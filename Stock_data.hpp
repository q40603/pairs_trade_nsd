#include <iostream>
#include <vector>
#include <map>
using namespace std;
class Stock_data
{
public:
    Stock_data() = default;;
    ~Stock_data() = default;;
    void read_csv(std::string _file);

    vector<float> operator() (string _id) const;
    friend ostream& operator<<(ostream &print, const Stock_data &d);
    

private:
   vector<vector<float> > _data;
   map<string, int> _company;
};