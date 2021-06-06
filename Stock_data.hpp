#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string.h>
#include <sstream>
#include <set>
#include <algorithm>
#include <math.h>
#include <fstream>
#include "./URT/include/URT.hpp"

using namespace std;
class Stock_data
{
public:
    Stock_data() = default;
    ~Stock_data() = default;
    Stock_data &operator=(Stock_data const &other);
    Stock_data& read_csv(string _file);
    Stock_data& drop_empty();
    Stock_data& to_log();
    Stock_data& drop_stationary();
    Stock_data& keep(int start, int end);
    vector<string> get_company_list();
    vector<vector<vector<double>>> get_pairs();
    vector<double> operator() (string _id) const;
    
    
public:
   map<string, vector<double> > _data;
};

Stock_data &Stock_data::operator=(Stock_data const &other) {
    if (this == &other) {
        return *this;
    }
    _data = other._data;
    return *this;
}

Stock_data& Stock_data::read_csv(string _filename){

    ifstream myFile(_filename);
    map<int, string> _company;
    if(!myFile.is_open()) throw runtime_error("Could not open file");
    string line, colname;
    char delim = ',' ;
    string val;
    int colIdx;

    // Read the column names
    if(myFile.good())
    {
        // Extract the first line in the file
        getline(myFile, line);

        // Create a stringstream from line
        stringstream ss(line);

        // Extract each column name
        int i = 0;
        while(getline(ss, colname, ',')){
            _company.insert(make_pair(i,colname));
            _data.insert(make_pair(colname,vector<double>()));
            i++;
        }
    }
    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        // Keep track of the current column index
        colIdx = 0;
        while (std::getline (ss, val, delim)){
            if(val=="") val="0";
            _data.at(_company.at(colIdx)).push_back(stof(val));
            colIdx++;
        }
    }
    return *this;
}

Stock_data& Stock_data::drop_empty(){
    map<string, vector<double> >::iterator it;
    vector<double>::iterator vit;
    vector<string> trash;
    for (it = _data.begin(); it != _data.end(); it++){
        vit = find(it->second.begin(), it->second.end(), 0.0);
        if (vit != it->second.end())
            trash.push_back(it->first);
    }
    while (!trash.empty()){
        _data.erase(trash.back());
        trash.pop_back();
    }
    return *this;

}

vector<double> Stock_data::operator() (string _id) const{
    if (_data.find(_id) == _data.end())
        throw runtime_error("key error : comapny not exists. ");
    return _data.at(_id);
}

Stock_data& Stock_data::to_log(){
    map<string, vector<double> >::iterator it;
    vector<double>::iterator vit;
    for (it = _data.begin(); it != _data.end(); it++){
        for(vit = it->second.begin(); vit != it->second.end(); vit++){
            *vit = log(*vit);
        }
    }
    return *this;
}


vector<string> Stock_data::get_company_list(){
    vector<string> res;
    for ( auto it = _data.begin(); it != _data.end(); ++it ){
        res.push_back(it->first);
    }
    return res;
}

vector<vector<vector<double>>> Stock_data::get_pairs(){
    vector<vector<vector<double>>> res;
    vector<string> clist = get_company_list();
    size_t l = clist.size();
    for (size_t i = 0 ; i < l ; i ++){
        for(size_t j = i+1 ; j < l ; j++ ){
            vector<vector<double>> _pair;
            _pair.push_back(_data.at(clist[i]));
            _pair.push_back(_data.at(clist[j]));
            res.push_back(_pair);
        }
    }

    return res;
}

Stock_data& Stock_data::drop_stationary(){
    map<string, vector<double> >::iterator it;
    vector<double>::iterator vit;
    vector<string> trash;
    double pval;
    for (it = _data.begin(); it != _data.end(); it++){
        urt::ADF<double> test(it->second, "AIC", "c", false);
        pval = test.pvalue();
        if(pval <= 0.05)
            trash.push_back(it->first);
    }    
    while (!trash.empty()){
        _data.erase(trash.back());
        trash.pop_back();
    }
    return *this;
}

Stock_data& Stock_data::keep(int start, int end){
    map<string, vector<double> >::iterator it;
    for (it = _data.begin(); it != _data.end(); it++){
        it->second = {it->second.begin() + start, it->second.begin() + end};
    }      
    return *this;  
}