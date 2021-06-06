#include<iostream>
#include "Stock_data.hpp"
#include "./URT/include/URT.hpp"

int main(){
    Stock_data data;
    vector<double> tmp;
    data = data.read_csv("./test.csv");
    data = data.drop_empty();
    data = data.to_log();
    
    vector<string> com_list = data.get_company_list();
    vector<vector<vector<double>>> pairs = data.get_pairs();
    // for(std::vector<string>::iterator it = com_list.begin(); it != com_list.end(); ++it){


    double stat;
    int i = 0;

    for(std::vector<string>::iterator it = com_list.begin(); it != com_list.end(); ++it){
        // cout<<*it<<endl;
        tmp = data(*it);
        urt::ADF<double> test(tmp, "AIC", "c", false);
        stat = test.statistic();
        // if (stat <= 0.05){
        cout << ++i << " " << *it << " " << stat <<endl;
            // test.show();
            
        // }
        // std::cout << std::fixed << std::setprecision(3);
        // cout<<std::setw(29)<<test.pvalue()<<endl;
        // test.show();

    }
    // tmp = data("2330");
    // int i = 0;
    // for (std::vector<double>::iterator it = tmp.begin() ; it !=tmp.end(); ++it){
    //     std::cout << i++ << ' ' << *it <<endl;
        
    // }
    // int max_lag = pow(12*(tmp.size())/100, 1/4);
    // cout<<max_lag<<endl;
    // urt::ADF<double> test(tmp, 7 , "c");
    // test.show();
    return 0;
}