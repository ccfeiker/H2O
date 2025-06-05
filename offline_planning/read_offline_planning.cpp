#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

int main() {
    const char* filepath = "/tmp/shared_offline_planning.bin";

    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open shared memory file\n";
        return 1;
    }

    void* map = mmap(nullptr, 8, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        std::cerr << "Failed to mmap\n";
        close(fd);
        return 1;
    }

    int k, w;
    std::memcpy(&k, map, sizeof(int));
    std::memcpy(&w, (char*)map + sizeof(int), sizeof(int));

    munmap(map, 8);
    close(fd);

    std::cout << "Read from shared memory: k = " << k << ", w = " << w << std::endl;

    return 0;
}

