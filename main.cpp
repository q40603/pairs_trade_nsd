#include<iostream>
#include "Stock_data.hpp"

int main(){
    Stock_data data;
    vector<float> tmp;
    data.read_csv("./test.csv");
    tmp = data("2330");
    for (std::vector<float>::iterator it = tmp.begin() ; it !=tmp.end(); ++it)
        std::cout << ' ' << *it <<endl;
    return 0;
}