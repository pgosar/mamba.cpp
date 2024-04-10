#ifndef MambaModel_hpp
#define MambaModel_hpp

#include <string>
#include <torch/torch.h>

class MambaModel : torch::nn::Module {

public:
    MambaModel();
    static void from_pretrained(const std::string&);
};

#endif