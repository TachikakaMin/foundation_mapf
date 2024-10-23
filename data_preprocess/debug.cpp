#include "H5Cpp.h"
#include <iostream>

int main() {
    std::string h5FilePath = "test.h5";

    try {
        H5::Exception::dontPrint();
        H5::H5File file(h5FilePath, H5F_ACC_TRUNC);
        H5::Group group = file.createGroup("/test_group");
        std::cout << "Successfully created HDF5 file and group." << std::endl;
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Exception: " << e.getDetailMsg() << std::endl;
    }

    return 0;
}
