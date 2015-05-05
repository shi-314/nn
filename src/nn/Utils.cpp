/**
* Utils
*
* This files provides some useful functions for the nn library.
*
* @author Shivan Taher
* @date 27.03.2009
*/

#include "Utils.h"

#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int randomInt(int min, int max) {
    return min + rand() * (max - min) / RAND_MAX;
}

float randomFloat(float min, float max) {
    return ((max - min) * ((float) rand() / RAND_MAX)) + min;
}

double randomDouble(double min, double max) {
    return ((max - min) * ((double) rand() / RAND_MAX)) + min;
}

vector<double> binToVec(string binStr, int minLen) {
    vector<double> v;
    for (unsigned int j = 0; j < minLen - binStr.length(); j++) {
        v.push_back(0);
    }
    for (unsigned int i = 0; i < binStr.length(); i++) {
        char c[2] = "";
        c[0] = binStr.c_str()[i];
        double num = (double) atoi((char *) c);
        v.push_back(num);
    }
    return v;
}

string doubleToString(double value) {
    stringstream ss;
    ss << value;
    return ss.str();
}

double roundDouble(double number, int digits) {
    // TODO: What the hell did I here?
    double v[] = {1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    return floor(number * v[digits] + 0.5) / v[digits];
}

int binVecToInt(vector<double> binVec) {
    char bin[16] = "";
    for (size_t i = 0; i < binVec.size(); i++)
        bin[i] = (int) roundDouble(binVec[i], 0) + '0';

    int b, k, m, n;
    int len, sum = 0;

    len = strlen(bin) - 1;
    for (k = 0; k <= len; k++) {
        n = (bin[k] - '0'); // char to numeric value
        if ((n > 1) || (n < 0)) {
            puts("\n\n ERROR! BINARY has only 1 and 0!\n");
            return (0);
        }
        for (b = 1, m = len; m > k; m--) {
            // 1 2 4 8 16 32 64 ... place-values, reversed here
            b *= 2;
        }
        // sum it up
        sum = sum + n * b;
        //printf("%d*%d + ",n,b);  // uncomment to show the way this works
    }
    return sum;
}
