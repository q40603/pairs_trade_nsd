#include<iostream>
#include "Stock_data.hpp"
#include "./URT/include/URT.hpp"

int main(){
    Stock_data data;
    vector<double> tmp;
    data.read_csv("./test.csv");
    data.drop_empty();
    data.to_log();
    tmp = data("2330");
    int i = 0;
    for (std::vector<double>::iterator it = tmp.begin() ; it !=tmp.end(); ++it){
        std::cout << i++ << ' ' << *it <<endl;
        
    }
    urt::ADF<double> test(tmp, "c");
    test.show();
    return 0;
}