/* Inference for Mamba model in pure C */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "args.hxx"
#include "flash_mem.hpp"
#include "mamba.hpp"
#include "math.hpp"
#include "tokenizer.hpp"
#include "util.hpp"

// ----------------------------------------------------------------------------
// header files change all the char arrays to std::strings

template <typename T>
void forward_layer(Mamba<T> *mamba, size_t l, EnhancedTensor<T> &hidden_state) {
  Config *p = &mamba->config;
  MambaWeights<T> *w = &mamba->weights;
  RunState<T> *s = &mamba->state;
  int dim = p->dim, d_inner = p->d_inner, d_conv = p->d_conv,
      d_state = p->d_state, dt_rank = p->dt_rank;
  EnhancedTensor<T> &dA = s->dA; // (d_inner, d_state)
  EnhancedTensor<T> &dB = s->dB; // (d_inner, d_state)
  EnhancedTensor<T> &y = s->y;   // (d_inner)
  float *const tempbuf = s->dequantized_buffer;

  // conv_state, ssm_state = self._get_states_from_cache(inference_params)
  EnhancedTensor<T> conv_state = s->conv_state.layer(l);
  EnhancedTensor<T> ssm_state = s->ssm_state.layer(l);

  // xz = self.in_proj(hidden_states)  # hidden_states: (dim), in_proj
  // (2*d_inner, dim), xz (2*d_inner)
  EnhancedTensor<T> layer_weight = w->in_proj.layer(l);
  matmul(s->xz, hidden_state, layer_weight, tempbuf, 2 * d_inner, dim);
  // x, z = xz.chunk(2, dim=-1)
  EnhancedTensor<T> x = s->xz.subset(0, d_inner); // x (d_inner)
  // TODO do we need to update the scale/zeropoint of the superset tensor?
  // or are they requantized before the read anyways...?

  // Conv step

  // conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
  // shift_matrix_left(conv_state, d_inner, d_conv);
  // conv_state[:, -1] = x
  // update_last_column(conv_state, x, d_inner, d_conv);
  shift_left_and_update_last(conv_state, x, tempbuf, d_inner, d_conv);

  // x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"),
  // dim=-1)
  elementwise_multiply(s->temp, conv_state, w->conv1d_weight.layer(l), tempbuf,
                       d_inner * d_conv);
  sum_along_last_dim(x, s->temp, tempbuf, d_inner, d_conv);
  // x = x + self.conv1d.bias
  elementwise_add(x, x, w->conv1d_bias.layer(l), tempbuf, d_inner);
  // x = F.silu(x)
  silu(x, tempbuf);

  // SSM step

  // x_db = self.x_proj(x)   # x_db (dt_rank+2*d_state)
  matmul(s->x_db, x, w->x_proj.layer(l), tempbuf, dt_rank + 2 * d_state,
         d_inner);
  // dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state],
  // dim=-1)
  EnhancedTensor<T> &dt = s->x_db;                   // dt (dt_rank)
  EnhancedTensor<T> B = s->x_db + dt_rank;           // B  (d_state)
  EnhancedTensor<T> C = s->x_db + dt_rank + d_state; // C  (d_state)

  // dt = self.dt_proj(dt)   # dt (dt_rank), dt_proj_weight (d_inner, dt_rank),
  // dt_proj_bias (d_inner)
  linear(s->dt, dt, w->dt_proj_weight.layer(l), w->dt_proj_bias.layer(l),
         tempbuf, d_inner, dt_rank);
  dt = s->dt; // NOTE: dt is now bigger: (d_inner) instead of (dt_rank)
  // dt = F.softplus(dt)
  softplus(dt, tempbuf);

  //  Discretize A and B
  // dA = torch.exp(torch.einsum("d,dn->dn", dt, self.A))   # A (d_inner,
  // d_state), dA (d_inner, d_state)
  broadcast_multiply(dA, dt, w->A.layer(l), tempbuf, d_inner, d_state);
  expf(dA, tempbuf);

  // dB = torch.einsum("d,n->dn", dt, B)    # dt (d_inner), B (d_state), dB
  // (d_inner, d_state)
  outer_product(dB, dt, B, tempbuf, d_inner, d_state);

  //  Update ssm_state
  // ssm_state.copy_(ssm_state * dA + rearrange(x, "d -> d 1") * dB)
  broadcast_multiply(s->temp, x, dB, tempbuf, d_inner, d_state);
  elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp, tempbuf,
                               d_inner * d_state);

  //  Compute y
  // y = torch.einsum("dn,n->d", ssm_state, C) # ssm_state (d_inner, d_state), C
  // (d_state), y (d_inner)
  rowwise_dot_product(y, ssm_state, C, tempbuf, d_inner, d_state);
  // y = y + self.D * x
  elementwise_multiply_and_add(y, w->D.layer(l), x, y, tempbuf, d_inner);

  // y = y * F.silu(z)  # (d_inner)
  silu(y, tempbuf);

  // hidden_state = self.out_proj(y)  # out_proj (dim, d_inner), hidden_state
  // (dim)
  matmul(hidden_state, y, w->out_proj.layer(l), tempbuf, dim, d_inner);

  s->conv_state.update_layer(l, conv_state);
  s->ssm_state.update_layer(l, ssm_state);
}

template <typename T> EnhancedTensor<T> &forward(Mamba<T> *mamba, int token) {
  // a few convenience variables
  Config *p = &mamba->config;
  MambaWeights<T> *w = &mamba->weights;
  RunState<T> *s = &mamba->state;
  int dim = p->dim;
  T *input_data = s->input.data();
  EnhancedTensor<T> &hidden_state = s->hidden_state;
  T *hidden_state_data = hidden_state.data();

  // copy the token embedding into x
  Tensor<T> content_row = w->token_embedding_table + token * dim;
  memcpy(input_data, content_row.data(), dim * sizeof(T));

  Tensor<T> inputTensor(w->token_embedding_table.scale(),
                        w->token_embedding_table.zeropoint(), input_data);

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // normalize the input
    rmsnorm(hidden_state, inputTensor, w->norm.layer(l), s->dequantized_buffer,
            dim);
    // forward this layer
    forward_layer(mamba, l, hidden_state);
    // residual connection back into hidden_state
    for (int i = 0; i < dim; i++) {
      s->dequantized_buffer[i] = input_data[i] + hidden_state_data[i];
    }

    hidden_state.requantize(s->dequantized_buffer);

    for (int i = 0; i < dim; i++) {
      // copy hidden_state back into input for the next layer
      input_data[i] = hidden_state_data[i];
    }
  }

  // final rmsnorm
  rmsnorm(hidden_state, hidden_state, w->final_norm, s->dequantized_buffer,
          dim);

  // classifier into logits
  matmul(s->logits, hidden_state, w->lm_head, s->dequantized_buffer,
         p->rounded_vocab_size, p->dim);
  return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop

template <typename T>
void generate(Mamba<T> *mamba, Tokenizer *tokenizer, Sampler *sampler,
              std::string &prompt, int steps, UserConfig userConfig) {
  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  const auto prompt_tokens = static_cast<int *>(
      malloc((prompt.length() + 3) * sizeof(int))); // +3 for '\0', BOS, EOS
  encode(tokenizer, prompt, 0, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // print the first token in the prompt
  if (num_prompt_tokens > 1) {
    char *piece = decode(tokenizer, EOS, prompt_tokens[0]);
    safe_printf(piece);
    fflush(stdout);
  }

  std::vector<int> prev_tokens;

  // start the main loop
  long start =
      0;    // used to time our code, only initialized after first iteration
  int next; // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence
  while (pos < steps) {
    // forward the model to get logits for the next token
    EnhancedTensor<T> &logits = forward(mamba, token);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits

      // modify for repetition penalty
      // gather on logits using previous output tokens
      // mult/divide on gathered "score"
      // scatter back into logits using above res

      // 1.0 does nothing.
      // values > 1 are pretty terrible and have high repetition
      // values < 1 help minimize repetition pretty well, but
      //   cause the quality of the response to suffer somewhat.
      //   For example, for sufficiently "low" penalty,
      //   "I" gets penalized too harshly, so the llm
      //   uses non-penalized alternatives like 1 or i.

      // Ideal seems to be penalty values slightly less than 1.

      // TODO this is needed, comment back-in after refactor integrity is
      // verified apply_repetition_penalty(logits, prev_tokens,
      //                          userConfig.repetition_penalty);

      next = sample(sampler, logits, mamba->state.dequantized_buffer);
    }
    pos++;

    // data-dependent terminating condition: the EOS token delimits sequences
    if (next == EOS) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    prev_tokens.push_back(next);

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first
  // iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n",
            (pos - 1) / (double)(end - start) * 1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, const size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != nullptr) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.
template <typename T>
void chat(Mamba<T> *mamba, Tokenizer *tokenizer, Sampler *sampler,
          const char *cli_user_prompt, const char *cli_system_prompt,
          int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
  int user_idx;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next;             // will store the next token in the sequence
  int token;            // stores the current token to feed into the model
  int pos = 0;          // position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == nullptr) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt,
                     sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        const char system_template[] =
            "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        snprintf(rendered_prompt, sizeof(rendered_prompt), system_template,
                 system_prompt, user_prompt);
      } else {
        const char user_template[] = "[INST] %s [/INST]";
        snprintf(rendered_prompt, sizeof(rendered_prompt), user_template,
                 user_prompt);
      }
      // encode the rendered prompt into tokens
      encode(tokenizer, rendered_prompt, 0, 0, prompt_tokens,
             &num_prompt_tokens);
      user_idx = 0; // reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the model next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS token ends the Assistant turn
    if (token == EOS) {
      user_turn = 1;
    }

    // forward the model to get logits for the next token
    EnhancedTensor<T> &logits = forward(mamba, token);
    next = sample(sampler, logits, mamba->state.dequantized_buffer);
    pos++;

    if (user_idx >= num_prompt_tokens && next != EOS) {
      // the Assistant is responding, so print its output
      char *piece = decode(tokenizer, token, next);
      safe_printf(
          piece); // same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (next == EOS) {
      printf("\n");
    }
  }
  printf("\n");
  free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
  std::cerr << "Usage:   run <checkpoint> [options]\n";
  std::cerr << "Example: run model.bin -n 256 -i \"Once upon a time\"\n";
  std::cerr << "Options:\n";
  std::cerr << "  -t <float>  temperature in [0,inf], default 1.0\n";
  std::cerr << "  -p <float>  p value in top-p (nucleus) sampling in [0,1] "
               "default 0.9\n";
  std::cerr << "  -s <int>    random seed, default time(NULL)\n";
  std::cerr << "  -n <int>    number of steps to run for, default 256\n";
  std::cerr << "  -i <string> input prompt\n";
  std::cerr << "  -z <string> optional path to custom tokenizer\n";
  std::cerr << "  -m <string> mode: generate|chat, default: generate\n";
  std::cerr << "  -y <string> (optional) system prompt in chat mode\n";
  std::cerr << "  --flashmem enable flash memory\n";
  std::cerr
      << "  --specdecoding enable speculative decoding with path to model\n";
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  args::ArgumentParser parser("Run the model with specified options.");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<float> temperatureFlag(
      parser, "float", "Temperature in [0,inf], default 1.0", {'t'}, 1.0f);
  args::ValueFlag<float> toppFlag(
      parser, "float",
      "P value in top-p (nucleus) sampling in [0,1], default 0.9", {'p'}, 0.9f);
  args::ValueFlag<int> stepsFlag(
      parser, "int", "Number of steps to run for, default 256", {'n'}, 256);
  args::ValueFlag<std::string> promptFlag(parser, "string", "Input prompt",
                                          {'i'}, "");
  args::ValueFlag<std::string> tokenizerPathFlag(
      parser, "string", "Optional path to custom tokenizer", {'z'},
      "models/tokenizer.bin");
  args::ValueFlag<std::string> modeFlag(
      parser, "string", "Mode: generate|chat, default: generate", {'m'},
      "generate");
  args::ValueFlag<std::string> systemPromptFlag(
      parser, "string", "System prompt in chat mode (optional)", {'y'});
  args::ValueFlag<float> repetitionPenaltyFlag(
      parser, "float", "Repetition penalty, default 1.0", {'r'}, 1.0f);
  args::Flag flashmemFlag(parser, "flashmem", "Enable flash memory",
                          {"flashmem"});
  args::ValueFlag<std::string> specDecodingFlag(
      parser, "specdecoding", "Path to the model for speculative decoding",
      {"specdecoding"});

  args::Positional<std::string> modelPath(parser, "checkpoint",
                                          "Path to the model checkpoint");

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  std::string model_path = args::get(modelPath);
  float temperature = args::get(temperatureFlag);
  float topp = args::get(toppFlag);
  int steps = args::get(stepsFlag);
  std::string prompt = args::get(promptFlag);
  std::string tokenizer_path = args::get(tokenizerPathFlag);
  std::string mode = args::get(modeFlag);
  std::string system_prompt =
      systemPromptFlag ? args::get(systemPromptFlag) : "";
  bool flashmem = flashmemFlag;
  std::string specDecodingModelPath =
      specDecodingFlag ? args::get(specDecodingFlag) : "";
  UserConfig userConfig;
  userConfig.repetition_penalty = args::get(repetitionPenaltyFlag);

  // Validating parameters
  if (temperature < 0.0f)
    temperature = 0.0f;
  if (topp < 0.0f || topp > 1.0f)
    topp = 0.9f;
  if (steps < 0)
    steps = 0;

  std::ifstream file(model_path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Couldn't open file " << model_path << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // get the file size
  Config *config = (Config *)malloc(sizeof(Config));
  file.seekg(0, std::ios::beg);

  // todo adjust dimensions to account for activation bool tensor
  // read the config
  // don't read rounded vocab size; that's computed here
  if (!file.read(reinterpret_cast<char *>(config), sizeof(Config))) {
    std::exit(EXIT_FAILURE);
  }

  if (config->vocab_size % 8 != 0) {
    config->rounded_vocab_size =
        config->vocab_size + (8 - (config->vocab_size % 8));
  } else {
    config->rounded_vocab_size = config->vocab_size;
  }

  file.close();

  if (flashmem)
    fprintf(stderr, "Flash memory enabled\n");
  if (specDecodingFlag)
    fprintf(stderr, "Speculative decoding enabled\n");
  // load the model using the model.bin file
  if (config->num_bits == 32) {
    Mamba<float> mamba;
    Mamba<float> aux_mamba; // one for spec decoding

    load_model(&mamba, model_path);
    load_model(&aux_mamba, specDecodingModelPath);

    // print the config
    fprintf(stderr,
            "config: vocab_size=%d (%d), n_layers=%d, dim=%d, d_inner=%d, "
            "dt_rank=%d, d_state=%d, d_conv=%d, bits=%d\n",
            mamba.config.vocab_size, mamba.config.rounded_vocab_size,
            mamba.config.n_layers, mamba.config.dim, mamba.config.d_inner,
            mamba.config.dt_rank, mamba.config.d_state, mamba.config.d_conv,
            mamba.config.num_bits);

    if (steps == 0)
      steps = 256; // override to default len if 0

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, mamba.config.vocab_size, temperature, topp);

    // run!
    if (mode.compare("generate") == 0) {
      generate(&mamba, &tokenizer, &sampler, prompt, steps, userConfig);
    } else if (mode.compare("chat") == 0) {
      chat(&mamba, &tokenizer, &sampler, prompt.c_str(), system_prompt.c_str(),
           steps);
    } else {
      fprintf(stderr, "unknown mode: %s\n", mode.c_str());
      error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
  } else if (config->num_bits == 16) {
    Mamba<int16_t> mamba;
    Mamba<int16_t> aux_mamba; // one for spec decoding

    load_model(&mamba, model_path);
    load_model(&aux_mamba, specDecodingModelPath);

    load_model(&mamba, model_path);

    // print the config
    fprintf(stderr,
            "config: vocab_size=%d (%d), n_layers=%d, dim=%d, d_inner=%d, "
            "dt_rank=%d, d_state=%d, d_conv=%d, bits=%d\n",
            mamba.config.vocab_size, mamba.config.rounded_vocab_size,
            mamba.config.n_layers, mamba.config.dim, mamba.config.d_inner,
            mamba.config.dt_rank, mamba.config.d_state, mamba.config.d_conv,
            mamba.config.num_bits);

    if (steps == 0)
      steps = 256; // override to default len if 0

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, mamba.config.vocab_size, temperature, topp);

    // run!
    if (mode.compare("generate") == 0) {
      generate(&mamba, &tokenizer, &sampler, prompt, steps, userConfig);
    } else if (mode.compare("chat") == 0) {
      chat(&mamba, &tokenizer, &sampler, prompt.c_str(), system_prompt.c_str(),
           steps);
    } else {
      fprintf(stderr, "unknown mode: %s\n", mode.c_str());
      error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
  } else if (config->num_bits == 8) {
    Mamba<int8_t> mamba;
    Mamba<int8_t> aux_mamba; // one for spec decoding
    load_model(&mamba, model_path);
    load_model(&aux_mamba, specDecodingModelPath);

    // print the config
    fprintf(stderr,
            "config: vocab_size=%d (%d), n_layers=%d, dim=%d, d_inner=%d, "
            "dt_rank=%d, d_state=%d, d_conv=%d, bits=%d\n",
            mamba.config.vocab_size, mamba.config.rounded_vocab_size,
            mamba.config.n_layers, mamba.config.dim, mamba.config.d_inner,
            mamba.config.dt_rank, mamba.config.d_state, mamba.config.d_conv,
            mamba.config.num_bits);

    if (steps == 0)
      steps = 256; // override to default len if 0

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, mamba.config.vocab_size, temperature, topp);

    // run!
    if (mode.compare("generate") == 0) {
      generate(&mamba, &tokenizer, &sampler, prompt, steps, userConfig);
    } else if (mode.compare("chat") == 0) {
      chat(&mamba, &tokenizer, &sampler, prompt.c_str(), system_prompt.c_str(),
           steps);
    } else {
      fprintf(stderr, "unknown mode: %s\n", mode.c_str());
      error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
  }

  return 0;
}
#endif
