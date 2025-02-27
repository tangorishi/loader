#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BUFFER_SIZE 1024 * 1024  // 1 MB
#define TOTAL_SIZE 1024 * 1024 * 1024 * 2LL  // 2 GB

double get_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main() {
    const char* tmp_file = "test.bin.tmp";
    const char* real_file = "test.bin";

    char buffer[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        buffer[i] = 'A';  // Fill with character 'A'
    }

    double start, end;

    FILE* fp = fopen(tmp_file, "w+b");
    if (fp) {
        start = get_time();

        for (long long i = 0; i < TOTAL_SIZE; i += BUFFER_SIZE) {
            if (fwrite(buffer, BUFFER_SIZE, 1, fp) != 1) {
                perror("Error writing to file");
                fclose(fp);
                return 1;
            }
        }

        fclose(fp);

        end = get_time();
        printf("Time taken for writing: %f seconds\n", end - start);

        start = get_time();

        if (rename(tmp_file, real_file) != 0) {
            perror("Error renaming file");
            return 1;
        }

        end = get_time();
        printf("Time taken for renaming: %f seconds\n", end - start);

    } else {
        perror("Error opening file");
        return 1;
    }

    printf("File operation successful.\n");
    return 0;
}
