#include <fstream>
#include "Stock_data.hpp"
#include <string.h>
#include <sstream>
#include <set>
// using namespace std;

void Stock_data::read_csv(string _filename){

    ifstream myFile(_filename);

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
            _company.insert(make_pair(colname,i));
            _data.push_back(vector<float>());
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
            _data[colIdx].push_back(stof(val));
            colIdx++;
        }
    }
}

vector<float> Stock_data::operator() (string _id) const{
    if (_company.find(_id) == _company.end())
        throw runtime_error("key error : comapny not exists. ");
    return _data[_company.at(_id)];
}
// ostream & operator<<(ostream &output, const Stock_data &d){
//     for (std::vector<int>::iterator it = d._data.begin() ; it != myvector.end(); ++it)
//         std::cout << ' ' << *it;
// }