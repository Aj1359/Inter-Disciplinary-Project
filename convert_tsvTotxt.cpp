#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

int main() {
    ifstream input("ca-AstroPh_inc.tsv");     // change to your tsv file name
    ofstream output("ca-AstroPh_inc.txt");   // output file

    if (!input.is_open()) {
        cout << "Error opening input file\n";
        return 1;
    }

    string line;

    while (getline(input, line)) {
        stringstream ss(line);
        string a, b;

        getline(ss, a, '\t');
        getline(ss, b, '\t');

        output << a << " " << b << endl;
    }

    input.close();
    output.close();

    cout << "Conversion complete\n";
    return 0;
}