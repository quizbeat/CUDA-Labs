#include <iostream>

int main() {

    int n;
    std::cin >> n;

    for (int i = 0; i < n; i++) {
        float value;
        std::cin >> value;
        std::cout << i << "\t: " << value << std::endl;
    }

    return 0;
}
