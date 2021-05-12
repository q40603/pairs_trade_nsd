#include<iostream>
#include "Stock_data.hpp"

int main(){
    Stock_data data;
    vector<float> tmp;
    data.read_csv("./test.csv");
    data.drop_empty();
    tmp = data("2330");
    int i = 0;
    for (std::vector<float>::iterator it = tmp.begin() ; it !=tmp.end(); ++it){
        std::cout << i++ << ' ' << *it <<endl;
        
    }
    return 0;
}