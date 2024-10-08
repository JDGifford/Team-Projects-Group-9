

#include <iostream>
#include <string.h>

#include "SimpleSerial.h"
#include <ctime>
#include <chrono>
#include <fstream>

int main(int argc, char const *argv[])
{

    std::string com_port = "\\\\.\\";
    com_port.append(argv[1]);

    std::cout << argv[1];

    DWORD COM_BAUD_RATE = CBR_115200;
    SimpleSerial Serial(com_port, COM_BAUD_RATE);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::ofstream file;
    file.open("output.txt");
    while (true)
    {
        if (Serial.IsConnected())
        {
            int reply_wait_time = 7;
            std::string syntax_type = "json";

            std::string incoming = Serial.ReadSerialPort(reply_wait_time, syntax_type);

            file << incoming;
        }

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
        if (ms.count() > 60000)
        {
            break;
        }
    }
    file.close();

    return 0;
}
