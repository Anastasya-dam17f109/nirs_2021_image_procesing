#pragma once
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/random.hpp>
#include <boost/math/distributions/rayleigh.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <atlimage.h>
#include <atlstr.h.>

using namespace boost::math;
using  namespace std;

const double pi = boost::math::constants::pi<double>();


struct target {
	int x;
	int y;
	int size;
	int brightness;
	int mix_type;
};

class mix_img_obj
{
	double** mixture_image = nullptr;
	double *   re_mix_shift = nullptr;
	double *   re_mix_scale = nullptr;
	LPCTSTR * mask_list;
	string mixture_type = "";
	string filename_gen_image = "D:\\generated_image.txt";
	CString item_type = L"item0";
	target* targs;
	std::ofstream out;
	unsigned image_len_x = 110;
	unsigned image_len_y = 110;
	unsigned class_amount = 1;
	unsigned amount_trg = 1;
	unsigned min_targ_size = 5;
	unsigned backg_size = 110;


public:
	mix_img_obj() {
	};
	mix_img_obj(int img_size, string mix_t, int amount_targets, int classes, bool file_flag);
	/*mix_img_obj(mix_img_obj& obj){

		mixture_image = new double

	}*/
	void     img_generator();
	void load_from_bitmap_item1();
	void load_from_bitmap_item2();
	void load_from_bitmap_item3();
	void     print_results();
	int      mean(double**);
	string   get_filename();
	double** get_image();
	std::pair<int, int> get_image_len();
	target*  get_targets();
	unsigned get_min_targ_size();
	double get_bask_shift() {
		return re_mix_shift[0];
	}
	double get_bask_scale() {
		return re_mix_scale[0];
	}
	double* get_shift() {
		return re_mix_shift;
	}
	double* get_scale() {
		return re_mix_scale;
	}
	string get_mixture_type() {
		return mixture_type;
	}
	int  get_class_amount() {
		return class_amount;
	}
	LPCTSTR * get_mask_list() {
		return mask_list;
	}
	CString get_item_type() {
		return item_type;
	}
	~mix_img_obj();
};



