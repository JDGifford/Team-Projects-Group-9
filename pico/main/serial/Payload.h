
#include <stdlib.h>
#include <cstddef>
#include <string>

#define PayloadSize 1024

class Payload
{
private:
    /* data */
public:
    Payload(/* args */);
    ~Payload();

    int data[PayloadSize];
    std::string imageName;
    int sequence;
};

Payload::Payload(/* args */)
{
}

Payload::~Payload()
{
}
