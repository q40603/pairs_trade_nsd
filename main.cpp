#include<iostream>
#include "Stock_data.hpp"

int main(){
    Stock_data data;
    data.read_csv("./test.csv");
    cout<<data("2331")[1];
    return 0;
}