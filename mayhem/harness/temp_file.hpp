#pragma once
#include <stdlib.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <string>

// Writes fuzz input to a unique file with the given suffix and removes it on destruction.
class TempFile {
public:
    TempFile(const uint8_t* data, size_t size, const char* suffix) {
        const char* dir = getenv("TMPDIR");
        std::string tmpl = std::string(dir && *dir ? dir : "/tmp") + "/mayhem-fuzz-XXXXXX" + suffix;
        std::string buf = tmpl;
        int fd = mkstemps(&buf[0], static_cast<int>(std::char_traits<char>::length(suffix)));
        if (fd < 0) {
            return;
        }
        size_t off = 0;
        while (off < size) {
            ssize_t n = write(fd, data + off, size - off);
            if (n <= 0) {
                break;
            }
            off += static_cast<size_t>(n);
        }
        close(fd);
        if (off == size) {
            m_path = buf;
        } else {
            unlink(buf.c_str());
        }
    }
    ~TempFile() {
        if (!m_path.empty()) {
            unlink(m_path.c_str());
        }
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    bool ok() const { return !m_path.empty(); }
    const std::string& path() const { return m_path; }

private:
    std::string m_path;
};
