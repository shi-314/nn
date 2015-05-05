/**
* Utils
*
* This files provides some useful functions for the nn library.
*
* @author Shivan Taher
* @date 27.03.2009
*/

#ifndef _UTILS_H
#define _UTILS_H

#include <bitset>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

int randomInt(int min, int max);

float randomFloat(float min, float max);

double randomDouble(double min, double max);

/**
* Converts the given data of type T to a string with their binary
* representations.
*/
template<typename T>
std::string toBin(const T &value) {
    const std::bitset<std::numeric_limits<T>::digits + 1> bs(value);
    const std::string s(bs.to_string());
    const std::string::size_type pos(s.find_first_not_of('0'));
    return pos == std::string::npos ? "0" : s.substr(pos);
}

/**
* Converts a string with binary values to a vector of doubles with their
* values.
*/
vector<double> binToVec(string binStr, int minLen);

string doubleToString(double value);

double roundDouble(double number, int digits);

int binVecToInt(vector<double> binVec);

#endif
