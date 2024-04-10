#include "MambaModel.hpp"

#include <stdlib.h>
#include <stdio.h>

void MambaModel::from_pretrained(const std::string& filepath) {
    FILE* fp = fopen(filepath.c_str(), "rb");

    if(fp) {
        size_t N;
        fread(&N, sizeof(unsigned long), 1, fp);

        size_t size = N+1;
        char* meta = (char*) malloc(size * sizeof(char));
        fread(meta, sizeof(char), size, fp);
        meta[N] = '\0';

        printf("%ld\n", N);
        printf("%ls\n", meta);

        free(meta);
        fclose(fp);
    } else {
        printf("Model file not found\n");
    }
}