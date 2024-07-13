/* Inference for Mamba model in pure C */

#include <fcntl.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "flash_mem.hpp"
#include "mamba.hpp"
#include "math.hpp"
#include "tokenizer.hpp"
#include "util.hpp"

// ----------------------------------------------------------------------------
// TODO: get rid of all the inlines by separating the hpp files into proper
// header files change all the char arrays to std::strings

template <typename T>
void forward_layer(Mamba<T> *mamba, size_t l, T *hidden_state) {
  Config *p = &mamba->config;
  MambaWeights<T> *w = &mamba->weights;
  RunState<T> *s = &mamba->state;
  int dim = p->dim, d_inner = p->d_inner, d_conv = p->d_conv,
      d_state = p->d_state, dt_rank = p->dt_rank;
  T *dA = s->dA; // (d_inner, d_state)
  T *dB = s->dB; // (d_inner, d_state)
  T *y = s->y;   // (d_inner)

  // conv_state, ssm_state = self._get_states_from_cache(inference_params)
  T *conv_state = s->conv_state + l * d_inner * d_conv;
  T *ssm_state = s->ssm_state + l * d_inner * d_state;

  // xz = self.in_proj(hidden_states)  # hidden_states: (dim), in_proj
  // (2*d_inner, dim), xz (2*d_inner)
  matmul(s->xz, hidden_state, w->in_proj + l * 2 * d_inner * dim, 2 * d_inner,
         dim);
  // x, z = xz.chunk(2, dim=-1)
  T *x = s->xz;           // x (d_inner)
  T *z = s->xz + d_inner; // z (d_inner)

  // Conv step

  // conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
  shift_matrix_left(conv_state, d_inner, d_conv);
  // conv_state[:, -1] = x
  update_last_column(conv_state, x, d_inner, d_conv);
  // x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"),
  // dim=-1)
  elementwise_multiply(s->temp, conv_state,
                       w->conv1d_weight + l * d_inner * d_conv,
                       d_inner * d_conv);
  sum_along_last_dim(x, s->temp, d_inner, d_conv);
  // x = x + self.conv1d.bias
  elementwise_add(x, x, w->conv1d_bias + l * d_inner, d_inner);
  // x = F.silu(x)
  for (int i = 0; i < d_inner; i++) {
    x[i] = silu(x[i]);
  }

  // SSM step

  // x_db = self.x_proj(x)   # x_db (dt_rank+2*d_state)
  matmul(s->x_db, x, w->x_proj + l * (dt_rank + 2 * d_state) * d_inner,
         dt_rank + 2 * d_state, d_inner);
  // dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state],
  // dim=-1)
  T *dt = s->x_db;                    // dt (dt_rank)
  T *B = s->x_db + dt_rank;           // B  (d_state)
  T *C = s->x_db + dt_rank + d_state; // C  (d_state)

  // dt = self.dt_proj(dt)   # dt (dt_rank), dt_proj_weight (d_inner, dt_rank),
  // dt_proj_bias (d_inner)
  linear(s->dt, dt, w->dt_proj_weight + l * d_inner * dt_rank,
         w->dt_proj_bias + l * d_inner, d_inner, dt_rank);
  dt = s->dt; // NOTE: dt is now bigger: (d_inner) instead of (dt_rank)
  // dt = F.softplus(dt)
  for (int i = 0; i < d_inner; i++) {
    dt[i] = softplus(dt[i]);
  }

  //  Discretize A and B
  // dA = torch.exp(torch.einsum("d,dn->dn", dt, self.A))   # A (d_inner,
  // d_state), dA (d_inner, d_state)
  broadcast_multiply(dA, dt, w->A + l * d_inner * d_state, d_inner, d_state);
  for (int i = 0; i < d_inner * d_state; i++) {
    dA[i] = expf(dA[i]);
  }
  // dB = torch.einsum("d,n->dn", dt, B)    # dt (d_inner), B (d_state), dB
  // (d_inner, d_state)
  outer_product(dB, dt, B, d_inner, d_state);

  //  Update ssm_state
  // ssm_state.copy_(ssm_state * dA + rearrange(x, "d -> d 1") * dB)
  broadcast_multiply(s->temp, x, dB, d_inner, d_state);
  elementwise_multiply_and_add(ssm_state, ssm_state, dA, s->temp,
                               d_inner * d_state);

  //  Compute y
  // y = torch.einsum("dn,n->d", ssm_state, C) # ssm_state (d_inner, d_state), C
  // (d_state), y (d_inner)
  rowwise_dot_product(y, ssm_state, C, d_inner, d_state);
  // y = y + self.D * x
  elementwise_multiply_and_add(y, w->D + l * d_inner, x, y, d_inner);
  // y = y * F.silu(z)  # (d_inner)
  for (int i = 0; i < d_inner; i++) {
    y[i] = y[i] * silu(z[i]);
  }

  // hidden_state = self.out_proj(y)  # out_proj (dim, d_inner), hidden_state
  // (dim)
  matmul(hidden_state, y, w->out_proj + l * dim * d_inner, dim, d_inner);
}

template <typename T> T *forward(Mamba<T> *mamba, int token) {
  // a few convenience variables
  Config *p = &mamba->config;
  MambaWeights<T> *w = &mamba->weights;
  RunState<T> *s = &mamba->state;
  int dim = p->dim;
  T *input = s->input;
  T *hidden_state = s->hidden_state;

  // copy the token embedding into x
  T *content_row = w->token_embedding_table + token * dim;
  memcpy(input, content_row, dim * sizeof(T));

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {
    // normalize the input
    rmsnorm(hidden_state, input, w->norm + l * dim, dim);
    // forward this layer
    forward_layer(mamba, l, hidden_state);
    // residual connection back into hidden_state
    for (int i = 0; i < dim; i++) {
      hidden_state[i] += input[i];
      // copy hidden_state back into input for the next layer
      input[i] = hidden_state[i];
    }
  }

  // final rmsnorm
  rmsnorm(hidden_state, hidden_state, w->final_norm, dim);

  // classifier into logits
  matmul(s->logits, hidden_state, w->lm_head, p->rounded_vocab_size, p->dim);
  return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop

template <typename T>
void generate(Mamba<T> *mamba, Tokenizer *tokenizer, Sampler *sampler,
              char *prompt, int steps, UserConfig userConfig) {
  if (prompt == NULL) {
    prompt = "";
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) *
                                     sizeof(int)); // +3 for '\0', BOS, EOS
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
    T *logits = forward(mamba, token);

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
      apply_repetition_penalty(logits, prev_tokens,
                               userConfig.repetition_penalty);

      next = sample(sampler, logits);
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

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
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
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

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
        if (cli_system_prompt == NULL) {
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
    T *logits = forward(mamba, token);
    next = sample(sampler, logits);
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
  fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
  fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] "
                  "default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

  // default parameters
  char *model_path = NULL; // e.g. out/model.bin
  char *tokenizer_path = "models/tokenizer.bin";
  float temperature =
      1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp =
      0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;         // number of steps to run for
  char *prompt = NULL;     // prompt string
  char *mode = "generate"; // generate|chat
  char *system_prompt =
      NULL; // the (optional) system prompt to use in chat mode

  // Build additional config from cmdline
  UserConfig userConfig;
  userConfig.repetition_penalty = 1.0;

  // poor man's C argparse so we can override the defaults above from the
  // command line
  if (argc >= 2) {
    model_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) {
      error_usage();
    } // must have arg after flag
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else if (argv[i][1] == 'r') {
      userConfig.repetition_penalty = atof(argv[i + 1]);
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (temperature < 0.0)
    temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp)
    topp = 0.9;
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
  // load the model using the model.bin file
  if (config->num_bits == 32) {
    Mamba<float> mamba;

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
    if (strcmp(mode, "generate") == 0) {
      generate(&mamba, &tokenizer, &sampler, prompt, steps, userConfig);
    } else if (strcmp(mode, "chat") == 0) {
      chat(&mamba, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
      fprintf(stderr, "unknown mode: %s\n", mode);
      error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
  } else if (config->num_bits == 16) {
    Mamba<int16_t> mamba;

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
    if (strcmp(mode, "generate") == 0) {
      generate(&mamba, &tokenizer, &sampler, prompt, steps, userConfig);
    } else if (strcmp(mode, "chat") == 0) {
      chat(&mamba, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
      fprintf(stderr, "unknown mode: %s\n", mode);
      error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_model(&mamba);
  } else if (config->num_bits == 8) {
    Mamba<int8_t> mamba;
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
    if (strcmp(mode, "generate") == 0) {
      generate(&mamba, &tokenizer, &sampler, prompt, steps, userConfig);
    } else if (strcmp(mode, "chat") == 0) {
      chat(&mamba, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
      fprintf(stderr, "unknown mode: %s\n", mode);
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
