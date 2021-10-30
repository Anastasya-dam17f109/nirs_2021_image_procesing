#include "pch.h"
#include "mixture_handler.h"


mixture_handler::mixture_handler(mix_img_obj* img, unsigned h_classes, double acc)
{
	gen_image = img;
	my_picture = gen_image->get_image();
	mixture_type = gen_image->get_mixture_type();
	hyp_cl_amount = h_classes;
	accuracy = acc;
	img_l_x = gen_image->get_image_len().first;
	img_l_y = gen_image->get_image_len().second;
	window_size = 10;
	min_trg_size = gen_image->get_min_targ_size();
	cout << endl;
	cout << " mix partition by " << h_classes << " components:" << endl;
	mixture_inicalization();
	/*if (mixture_type == "normal")
		SEMalgorithm_opMP_normal_spec();
	else {
		if(mixture_type == "rayleigh")
			SEMalgorithm_opMP_rayleigh_spec();
	}*/
	
	//create_mesh_opMP();
	/*SEMalgorithm();
	create_mesh();*/
	//optimal_redraw();
	optimal_redraw_opMP();
	kolmogorov_optimal_redraw_opMP();
	
	//detect_results();
	//BIC();
	//draw_graphics();

}

//инициализация параметров смеси

void mixture_handler::mixture_inicalization() {
	double last_shift = 0;
	double last_scale = 10;
	unsigned i, j;
	mix_shift = new double[hyp_cl_amount];
	mix_scale = new double[hyp_cl_amount];
	mix_weight = new double[hyp_cl_amount];
	// создаем массив-результат
	class_flag = new unsigned    *[img_l_x];
	for (i = 0; i < img_l_x; i++) {
		class_flag[i] = new unsigned[img_l_y];
		for (j = 0; j < img_l_y; j++)
			class_flag[i][j] = 0;
	}

	for (i = 0; i < hyp_cl_amount; i++) {
		if (i > 0) {
			last_shift = mix_shift[i - 1];
			last_scale = 1;
		}
		mix_shift[i] = pow(0.5, i + 1)*250.0 + last_shift;
		if (hyp_cl_amount == 2)
			mix_shift[1] = 200;
		mix_scale[i] = (2.0*(i + 1))*last_scale;
		mix_weight[i] = 0.0;
	}
}

//

void mixture_handler::EMalgorithm() {
	unsigned block_size = img_l_x / thr_nmb;
	unsigned k = hyp_cl_amount;
	long double summ = 0;
	long double summ1 = 0;
	long double new_n = block_size * block_size;
	unsigned u_new_n = block_size * block_size;
	long double **re_comp_shifts = new long double*[amount_trg*amount_trg];
	long double **re_comp_scales = new long double*[amount_trg*amount_trg];
	long double **all_comp_shifts = new long double*[amount_trg*amount_trg];
	long double **all_comp_scales = new long double*[amount_trg*amount_trg];
	unsigned   **buf_comp_weights = new unsigned*[thr_nmb];
	double* class_center = new double[k];
	double max_buf = 0;
	unsigned v_size = 0;
	unsigned counter = 0;
	bool stop_flag = true;
	unsigned mesh_step = 10;
	unsigned i, t, r;

	for (i = 0; i < amount_trg*amount_trg; ++i) {
		re_comp_shifts[i] = new long double[k];
		re_comp_scales[i] = new long double[k];
		all_comp_shifts[i] = new long double[k];
		all_comp_scales[i] = new long double[k];
		for (t = 0; t < k; ++t) {
			re_comp_shifts[i][t] = 0;
			re_comp_scales[i][t] = 0;
			all_comp_shifts[i][t] = 0;
			all_comp_scales[i][t] = 0;
		}
	}

	auto mix_part_computation = [&](unsigned i) {
		long double summ = 0;
		double last_cur_max = 0;
		long double pix_buf = 0;

		double cur_max = 0;
		long double** new_g_ij = new long double*[new_n];
		long double** new_g_ij_0 = new long double*[new_n];
		long double * new_weights = new long double[k];
		long double * new_shifts = new long double[k];
		long double * new_scales = new long double[k];
		long double * buf_new_weights = new long double[k];
		long double * buf_new_shifts = new long double[k];
		bool stop_flag = true;
		double min_p = 289.0 / (2 * 110.0*110.0);
		double buf_max = 0;
		unsigned idx_i = 0;
		unsigned idx_j = 0;
		unsigned idx_max = 0;
		unsigned x_min, y_min, l, j, t;
		x_min = i * block_size;

		//double min_p = 0.005;

		for (l = 0; l < u_new_n; ++l) {
			new_g_ij[l] = new long double[k];
			new_g_ij_0[l] = new long double[k];
			for (t = 0; t < k; t++) {
				new_g_ij[l][t] = 0;
				new_g_ij_0[l][t] = 0;
			}
		}
		for (l = 0; l < k; ++l) {
			new_weights[l] = 0;
			new_shifts[l] = 0;
			new_scales[l] = 0;
			buf_new_weights[l] = 0;
			buf_new_shifts[l] = 0;
		}

		for (j = 0; j < amount_trg; ++j) {
			y_min = j * block_size;
			stop_flag = true;
			cur_max = 0;
			for (l = 0; l < k; ++l) {
				new_weights[l] = 0.5;
				new_shifts[l] = mix_shift[l];
				new_scales[l] = mix_scale[l];
			}
			for (l = 0; l < u_new_n; ++l) {
				for (t = 0; t < k; ++t)
					new_g_ij_0[l][t] = 0;
			}
			while (stop_flag) {
				for (l = 0; l < u_new_n; ++l) {
					summ = 0;
					idx_i = x_min + unsigned(l / block_size);
					idx_j = y_min + l % block_size;
					pix_buf = my_picture[idx_i][idx_j];
					for (t = 0; t < k; ++t) {
						if (new_weights[t] != 0
							&& new_scales[t] != 0)
							summ += new_weights[t] * (1 / (new_scales[t] * sqrt(2 * pi)))*exp(-((pix_buf
								- new_shifts[t])*(pix_buf - new_shifts[t])) / (2.0 * new_scales[t] * new_scales[t]));
					}
					for (t = 0; t < k; ++t) {
						if (l == 0) {
							buf_new_weights[t] = 0;
							buf_new_shifts[t] = 0;
						}
						if (new_weights[t] != 0
							&& new_scales[t] != 0) {
							new_g_ij[l][t] = new_weights[t] * (1 / (new_scales[t] * sqrt(2 * pi)*summ))*exp(-((pix_buf
								- new_shifts[t])*(pix_buf - new_shifts[t])) / (2.0 * new_scales[t] * new_scales[t]));
							buf_new_weights[t] += new_g_ij[l][t];
							buf_new_shifts[t] += new_g_ij[l][t] * pix_buf;
						}
						if (l == u_new_n - 1) {
							new_weights[t] = buf_new_weights[t] / (new_n);
							new_shifts[t] = buf_new_shifts[t] / (new_n*new_weights[t]);
						}
					}
				}

				for (t = 0; t < u_new_n; ++t) {
					for (l = 0; l < k; ++l) {
						if (t == 0)
							new_scales[l] = 0;
						idx_i = x_min + int(t / block_size);
						idx_j = y_min + t % block_size;
						pix_buf = my_picture[idx_i][idx_j];
						new_scales[l] += new_g_ij[t][l] * (pix_buf
							- new_shifts[l])*(pix_buf - new_shifts[l]);
						if (t == u_new_n - 1)
							new_scales[l] = sqrt(new_scales[l] / (new_n*new_weights[l]));
						if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
							cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
						new_g_ij_0[t][l] = new_g_ij[t][l];
					}
				}

				if (stop_flag != false) {
					if (cur_max != 0)
						last_cur_max = cur_max;
					(cur_max < accuracy) ? stop_flag = false : cur_max = 0;
				}
			}

			for (l = 0; l < k; ++l)
				new_weights[l] = 0;
			for (t = 0; t < new_n; ++t) {
				idx_i = x_min + unsigned(t / block_size);
				idx_j = y_min + t % block_size;
				buf_max = new_g_ij[t][0];
				idx_max = 0;
				for (l = 0; l < k; ++l) {
					if (buf_max < new_g_ij[t][l]) {
						buf_max = new_g_ij[t][l];
						idx_max = l;
					}
				}
				class_flag[idx_i][idx_j] = idx_max + 1;
				++new_weights[idx_max];
			}
			for (l = 0; l < k; ++l) {
				if (new_weights[l] / double(block_size*block_size) > min_p) {
					re_comp_shifts[i* amount_trg + j][l] = new_shifts[l];
					re_comp_scales[i* amount_trg + j][l] = new_scales[l];
				}
				all_comp_shifts[i* amount_trg + j][l] = new_shifts[l];
				all_comp_scales[i* amount_trg + j][l] = new_scales[l];
			}
		}
		for (t = 0; t < u_new_n; ++t) {
			delete[] new_g_ij[t];
			delete[] new_g_ij_0[t];
		}
		delete[] new_g_ij;
		delete[] new_g_ij_0;
		delete[] new_weights;
		delete[] new_shifts;
		delete[] new_scales;
		delete[] buf_new_weights;
		delete[] buf_new_shifts;
	};

	auto redraw = [&](unsigned i) {
		unsigned x_min = i * block_size;
		unsigned y_min, j, t, r, numb, new_numb;

		for (j = 0; j < amount_trg; ++j) {
			y_min = j * block_size;
			for (t = x_min; t < x_min + block_size; ++t) {
				for (r = y_min; r < y_min + block_size; ++r) {
					numb = class_flag[t][r] - 1;
					new_numb = all_comp_shifts[i*amount_trg][numb];
					class_flag[t][r] = new_numb + 1;
					buf_comp_weights[i][new_numb] += 1;
				}
			}
		}
	};

	auto begin1 = std::chrono::steady_clock::now();
	std::thread threadObj1(mix_part_computation, 0);
	std::thread threadObj2(mix_part_computation, 1);
	std::thread threadObj3(mix_part_computation, 2);
	std::thread threadObj4(mix_part_computation, 3);
	std::thread threadObj5(mix_part_computation, 4);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n" << "\n" << "\n";

	double left_bound, right_bound, remainder;
	auto begin2 = std::chrono::steady_clock::now();
	while (stop_flag) {
		v_size = 260 / mesh_step;
		dist_mesh = new  mesh_elem[v_size];
		for (i = 0; i < v_size; i++)
			dist_mesh[i].knot_elem = i * mesh_step;

		for (i = 0; i < amount_trg*amount_trg; ++i) {
			for (t = 0; t < k; ++t) {
				if (re_comp_shifts[i][t] != 0) {
					left_bound = trunc(re_comp_shifts[i][t] / double(mesh_step));
					right_bound = ceil(re_comp_shifts[i][t] / double(mesh_step));
					if (left_bound >= v_size) {
						left_bound = v_size - 1;
						right_bound = v_size;
					}
					remainder = re_comp_shifts[i][t] - left_bound * double(mesh_step);
					if (remainder < double(mesh_step) / 2.0 + 0.055) {
						dist_mesh[unsigned(left_bound)].rights.push_back(re_comp_shifts[i][t]);
						dist_mesh[unsigned(left_bound)].rights_SKO.push_back(re_comp_scales[i][t]);
						++dist_mesh[unsigned(left_bound)].amount;
					}
					else {
						if (unsigned(right_bound) != v_size) {
							dist_mesh[unsigned(right_bound)].lefts.push_back(re_comp_shifts[i][t]);
							dist_mesh[unsigned(right_bound)].lefts_SKO.push_back(re_comp_scales[i][t]);
							++dist_mesh[unsigned(right_bound)].amount;
						}
						else {
							dist_mesh[unsigned(left_bound)].rights.push_back(re_comp_shifts[i][t]);
							dist_mesh[unsigned(left_bound)].rights_SKO.push_back(re_comp_scales[i][t]);
							++dist_mesh[unsigned(left_bound)].amount;
						}
					}
				}
			}
		}

		counter = 0;
		for (i = 0; i < v_size; i++)
			(dist_mesh[i].amount != 0) ? ++counter : counter += 0;

		if (counter == k)
			stop_flag = false;
		else {
			if (counter > k) {
				int middle;
				for (i = 0; i < v_size - 1; ++i) {
					if (dist_mesh[i].amount != 0 &&
						dist_mesh[i + 1].amount != 0) {
						if (dist_mesh[i].lefts.size() > 0 &&
							dist_mesh[i].rights.size() > 0 &&
							dist_mesh[i + 1].lefts.size() > 0 &&
							dist_mesh[i + 1].rights.size() == 0) {
							r = 0;
							middle = (dist_mesh[i + 1].knot_elem + dist_mesh[i].knot_elem) / 2;
							for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++)
								if (abs(dist_mesh[i + 1].lefts[t] - middle) <= abs(dist_mesh[i + 1].lefts[t] - dist_mesh[i + 1].knot_elem))
									r++;
							if (dist_mesh[i + 1].lefts.size() - r <= r) {
								for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++) {
									dist_mesh[i].rights.push_back(dist_mesh[i + 1].lefts[t]);
									dist_mesh[i].rights_SKO.push_back(dist_mesh[i + 1].lefts_SKO[t]);
								}
								dist_mesh[i].amount += dist_mesh[i].rights.size();
								dist_mesh[i + 1].lefts.clear();
								dist_mesh[i + 1].lefts_SKO.clear();
								dist_mesh[i + 1].amount = 0;
								--counter;
							}
						}
						else {
							if (dist_mesh[i].lefts.size() == 0 &&
								dist_mesh[i].rights.size() > 0 &&
								dist_mesh[i + 1].lefts.size() > 0 &&
								dist_mesh[i + 1].rights.size() > 0) {
								r = 0;
								middle = (dist_mesh[i + 1].knot_elem + dist_mesh[i].knot_elem) / 2;
								for (t = 0; t < dist_mesh[i].rights.size(); t++)
									if (abs(dist_mesh[i].rights[t] - middle) <= abs(dist_mesh[i].rights[t] - dist_mesh[i].knot_elem))
										r++;
								if (dist_mesh[i].rights.size() - r <= r) {
									for (t = 0; t < dist_mesh[i].rights.size(); t++) {
										dist_mesh[i + 1].lefts.push_back(dist_mesh[i].rights[t]);
										dist_mesh[i + 1].lefts_SKO.push_back(dist_mesh[i].rights_SKO[t]);
									}
									dist_mesh[i + 1].amount += dist_mesh[i].rights.size();
									dist_mesh[i].rights.clear();
									dist_mesh[i].rights_SKO.clear();
									dist_mesh[i].amount = 0;
									--counter;
								}

							}
							else {
								if (dist_mesh[i].lefts.size() == 0 &&
									dist_mesh[i].rights.size() > 0 &&
									dist_mesh[i + 1].lefts.size() > 0 &&
									dist_mesh[i + 1].rights.size() == 0) {
									r = 0;
									middle = (dist_mesh[i + 1].knot_elem + dist_mesh[i].knot_elem) / 2;
									for (t = 0; t < dist_mesh[i].rights.size(); t++)
										if (abs(dist_mesh[i].rights[t] - middle) <= abs(dist_mesh[i].rights[t] - dist_mesh[i].knot_elem))
											r++;
									if (dist_mesh[i].rights.size() - r <= r) {
										for (t = 0; t < dist_mesh[i].rights.size(); t++) {
											dist_mesh[i + 1].lefts.push_back(dist_mesh[i].rights[t]);
											dist_mesh[i + 1].lefts_SKO.push_back(dist_mesh[i].rights_SKO[t]);
										}
										dist_mesh[i + 1].amount += dist_mesh[i].rights.size();
										dist_mesh[i].rights.clear();
										dist_mesh[i].rights_SKO.clear();
										dist_mesh[i].amount = 0;
										--counter;
									}
									else {
										r = 0;
										middle = (dist_mesh[i + 1].knot_elem + dist_mesh[i].knot_elem) / 2;
										for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++)
											if (abs(dist_mesh[i + 1].lefts[t] - middle) <= abs(dist_mesh[i + 1].lefts[t] - dist_mesh[i + 1].knot_elem))
												r++;
										if (dist_mesh[i + 1].lefts.size() - r <= r) {
											for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++) {
												dist_mesh[i].rights.push_back(dist_mesh[i + 1].lefts[t]);
												dist_mesh[i].rights_SKO.push_back(dist_mesh[i + 1].lefts_SKO[t]);
											}
											dist_mesh[i].amount += dist_mesh[i].rights.size();
											dist_mesh[i + 1].lefts.clear();
											dist_mesh[i + 1].lefts_SKO.clear();
											dist_mesh[i + 1].amount = 0;
											--counter;
										}
									}
								}
							}

						}

					}
					if (counter == k) {
						stop_flag = false;
						break;
					}
				}
				if (counter > k) {
					cout << "Houston, we have a problem! This is too much!" << endl;
					stop_flag = false;
				}
			}
			else {
				if (mesh_step > 1) {
					for (i = 0; i < v_size; i++) {
						dist_mesh[i].lefts.clear();
						dist_mesh[i].lefts_SKO.clear();
						dist_mesh[i].rights.clear();
						dist_mesh[i].rights_SKO.clear();
					}
					delete[] dist_mesh;
					mesh_step = mesh_step / 2;
				}
				else {
					cout << "Houston, we have a problem! This is not enough!" << endl;
					stop_flag = false;
				}
			}
		}
	}

	if (counter != k) {
		cout << " thought Mix components amount is absolutely wrong. Choose another model. redraw may have some mistakes. " << counter << endl;
		cout << " thought Mix components amount = " << hyp_cl_amount << ", finded  Mix components amount " << counter << endl;
		equal_hyp_flag = false;
		hyp_cl_amount = counter;
		delete[] mix_shift;
		delete[] mix_scale;
		delete[] mix_weight;
		mix_shift = new double[hyp_cl_amount];
		mix_scale = new double[hyp_cl_amount];
		mix_weight = new double[hyp_cl_amount];
		for (i = 0; i < hyp_cl_amount; ++i)
			mix_weight[i] = 0.0;
	}

	r = 0;
	for (i = 0; i < v_size; i++) {
		if (dist_mesh[i].amount != 0) {
			summ = 0;
			summ1 = 0;
			for (t = 0; t < dist_mesh[i].lefts.size(); ++t) {
				summ += dist_mesh[i].lefts[t];
				summ1 += dist_mesh[i].lefts_SKO[t];
			}
			for (t = 0; t < dist_mesh[i].rights.size(); ++t) {
				summ += dist_mesh[i].rights[t];
				summ1 += dist_mesh[i].rights_SKO[t];
			}
			mix_shift[r] = summ / double(dist_mesh[i].amount);
			mix_scale[r] = summ1 / double(dist_mesh[i].amount);
			++r;
		}
	}
	for (i = 0; i < amount_trg*amount_trg; ++i) {
		for (t = 0; t < k; ++t) {
			max_buf = 0;
			for (r = 0; r < counter; ++r) {
				summ = sqrt((all_comp_shifts[i][t] - mix_shift[r])*(all_comp_shifts[i][t] - mix_shift[r])
					+ (all_comp_scales[i][t] - mix_scale[r])*(all_comp_scales[i][t] - mix_scale[r]));
				if (r == 0) {
					max_buf = summ;
					summ1 = r;
				}
				else {
					if (max_buf > summ) {
						max_buf = summ;
						summ1 = r;
					}
				}
			}
			all_comp_shifts[i][t] = summ1;
		}
	}
	for (i = 0; i < thr_nmb; ++i) {
		buf_comp_weights[i] = new unsigned[hyp_cl_amount];
		for (t = 0; t < hyp_cl_amount; ++t)
			buf_comp_weights[i][t] = 0;
	}

	cout << "EM mix_shift values  1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << endl;
	cout << "EM mix_scale values 1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	cout << "EM mix_shift values  2 step:" << "\n";
	for (t = 0; t < amount_trg*amount_trg; ++t) {
		for (i = 0; i < k; i++)
			cout << re_comp_shifts[t][i] << "  ";
		cout << endl;
	}

	
	std::thread threadObj6(redraw, 0);
	std::thread threadObj7(redraw, 1);
	std::thread threadObj8(redraw, 2);
	std::thread threadObj9(redraw, 3);
	std::thread threadObj10(redraw, 4);
	threadObj6.join();
	threadObj7.join();
	threadObj8.join();
	threadObj9.join();
	threadObj10.join();

	for (i = 0; i < thr_nmb; ++i) {
		for (t = 0; t < hyp_cl_amount; ++t)
			mix_weight[t] += buf_comp_weights[i][t] / double(img_l_x*img_l_y);
	}



	for (i = 0; i < thr_nmb; ++i)
		delete[] buf_comp_weights[i];
	for (i = 0; i < amount_trg*amount_trg; ++i) {
		delete[] re_comp_shifts[i];
		delete[] re_comp_scales[i];
		delete[] all_comp_shifts[i];
		delete[] all_comp_scales[i];
	}
	delete[] re_comp_shifts;
	delete[] re_comp_scales;
	delete[] all_comp_shifts;
	delete[] all_comp_scales;
	delete[] class_center;
	delete[] buf_comp_weights;
	for (i = 0; i < v_size; i++) {
		dist_mesh[i].lefts.clear();
		dist_mesh[i].lefts_SKO.clear();
		dist_mesh[i].rights.clear();
		dist_mesh[i].rights_SKO.clear();
	}
	delete[] dist_mesh;

}

void mixture_handler::SEMalgorithm() {
	//window_size = window_size*2;
	unsigned block_size = img_l_x / thr_nmb;
	unsigned h_w = (window_size / 2);
	unsigned vert = block_size / h_w;
	unsigned horiz = img_l_x / h_w - 1;
	amount_trg = vert * horiz;
	unsigned k = hyp_cl_amount;
	long double summ = 0;
	long double summ1 = 0;
	long double new_n = window_size * window_size;
	const unsigned u_new_n = window_size * window_size;
	re_comp_shifts = new long double*[thr_nmb*amount_trg - horiz];
	re_comp_scales = new long double*[thr_nmb*amount_trg - horiz];
	double*  class_center = new double[k];
	double   max_buf = 0;
	unsigned v_size = 0;
	unsigned counter = 0;
	bool     stop_flag = true;
	unsigned mesh_step = 10;
	unsigned i, t, r;

	for (i = 0; i < thr_nmb*amount_trg - horiz; ++i) {
		re_comp_shifts[i] = new long double[k];
		re_comp_scales[i] = new long double[k];

		for (t = 0; t < k; ++t) {
			re_comp_shifts[i][t] = 0;
			re_comp_scales[i][t] = 0;
		}
	}

	auto mix_part_computation = [&](unsigned i) {
		long double   summ = 0;
		long double   pix_buf = 0;
		double        cur_max = 0;
		long double** new_g_ij = new long double*[new_n];
		long double** new_g_ij_0 = new long double*[new_n];
		unsigned**    y_i_j = new unsigned*[u_new_n];
		long double * new_weights = new long double[k];
		long double * new_shifts = new long double[k];
		long double * new_scales = new long double[k];

		bool          stop_flag = true;
		const double  sq_pi = sqrt(2 * pi);
		double        min_p = double(min_trg_size* min_trg_size) / double(window_size*window_size);
		double        buf_max = 0;
		unsigned      idx_i = 0;
		unsigned      idx_j = 0;
		unsigned      idx_max = 0;
		double val;
		unsigned x_min, y_min, l, j, t, r, i_lim, i_step, itr;
		double bound_d, bound_u, psumm;
		boost::random::uniform_01 <> dist_poly;
		boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };

		for (l = 0; l < u_new_n; ++l) {
			new_g_ij[l] = new long double[k];
			new_g_ij_0[l] = new long double[k];
			y_i_j[l] = new unsigned[k];
			psumm = 0;
			for (t = 0; t < k; t++) {
				val = dist_poly(generator);
				psumm += val;
				new_g_ij[l][t] = val;
				new_g_ij_0[l][t] = 0;
				y_i_j[l][t] = 0;
				if (l == 0) {
					new_weights[l] = 0;
					new_shifts[l] = 0;
					new_scales[l] = 0;
				}
			}
			for (t = 0; t < k; t++)
				new_g_ij[l][t] = new_g_ij[l][t] / psumm;
			//new_g_ij[l][t] = 1.0 / double(k);


		}


		i_lim = vert - 1;
		i_step = unsigned(window_size / 2);
		x_min = i * block_size - i_step;
		if (i == thr_nmb - 1)
			--i_lim;
		for (r = 0; r < i_lim; ++r) {
			x_min += i_step;
			for (j = 0; j < horiz; ++j) {
				y_min = j * i_step;
				stop_flag = true;
				cur_max = 0;
				itr = 0;
				while (stop_flag &&  itr < 300) {
					itr++;
					for (l = 0; l < u_new_n; ++l) {
						summ = 0;
						idx_i = x_min + l / window_size;
						idx_j = y_min + l % window_size;
						pix_buf = my_picture[idx_i][idx_j];
						bound_d = 0;
						bound_u = new_g_ij[l][0];
						val = dist_poly(generator);
						for (t = 0; t < k; ++t) {
							if (val < bound_u
								&& val >= bound_d) {
								y_i_j[l][t] = 1;
								new_weights[t] += 1;
								new_shifts[t] += pix_buf;
							}
							else
								y_i_j[l][t] = 0;
							bound_d += new_g_ij[l][t];
							if (t < k - 2)
								bound_u += new_g_ij[l][t + 1];
							else
								if (t == k - 2)
									bound_u = 1;
						}
					}

					for (t = 0; t < u_new_n; ++t) {
						for (l = 0; l < k; ++l) {
							if (new_weights[l] > 0) {
								idx_i = x_min + t / window_size;
								idx_j = y_min + t % window_size;
								pix_buf = my_picture[idx_i][idx_j];
								if (t == 0)
									new_shifts[l] = new_shifts[l] / new_weights[l];
								new_scales[l] += y_i_j[t][l] * (pix_buf
									- new_shifts[l])*(pix_buf - new_shifts[l]);
								if (t == u_new_n - 1) {
									new_scales[l] = sqrt(new_scales[l] / new_weights[l]);
									new_weights[l] = new_weights[l] / new_n;
								}
							}
							if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
								cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
							new_g_ij_0[t][l] = new_g_ij[t][l];
						}
					}
					for (l = 0; l < u_new_n; ++l) {
						idx_i = x_min + l / window_size;
						idx_j = y_min + l % window_size;
						summ = 0;
						pix_buf = my_picture[idx_i][idx_j];
						for (t = 0; t < k; ++t) {
							if (new_weights[t] >= 0
								&& new_scales[t] >= 0)
								summ += new_weights[t] * (1 / (new_scales[t] * sq_pi))*exp(-((pix_buf
									- new_shifts[t])*(pix_buf - new_shifts[t])) / (2.0 * new_scales[t] * new_scales[t]));
						}
						for (t = 0; t < k; ++t)
							if (new_weights[t] >= 0
								&& new_scales[t] >= 0)
								new_g_ij[l][t] = new_weights[t] * (1 / (new_scales[t] * sq_pi*summ))*exp(-((pix_buf
									- new_shifts[t])*(pix_buf - new_shifts[t])) / (2.0 * new_scales[t] * new_scales[t]));
							else
								new_g_ij[l][t] = 0;
					}
					if (stop_flag != false) {
						if (cur_max < accuracy)
							stop_flag = false;
						else {
							cur_max = 0;
							for (l = 0; l < k; ++l) {
								new_weights[l] = 0;
								new_shifts[l] = 0;
								new_scales[l] = 0;
							}
						}
					}
				}
				for (l = 0; l < k; ++l)
					new_weights[l] = 0;
				for (t = 0; t < u_new_n; ++t) {
					idx_i = x_min + t / window_size;
					idx_j = y_min + t % window_size;
					buf_max = new_g_ij[t][0];
					idx_max = 0;
					psumm = 0;
					for (l = 0; l < k; ++l) {
						if (buf_max < new_g_ij[t][l]) {
							buf_max = new_g_ij[t][l];
							idx_max = l;
						}
						new_g_ij_0[t][l] = 0;
						val = dist_poly(generator);
						psumm += val;
						new_g_ij[t][l] = val;
					}
					++new_weights[idx_max];
					for (l = 0; l < k; l++)
						new_g_ij[t][l] = new_g_ij[t][l] / psumm;
					//new_g_ij[t][l] = 1.0 / double(k);

				}
				for (l = 0; l < k; ++l) {
					if (new_weights[l] / double(window_size*window_size) > min_p
						&& new_scales[l] > 0) {
						re_comp_shifts[i* amount_trg + r * horiz + j][l] = new_shifts[l];
						re_comp_scales[i* amount_trg + r * horiz + j][l] = new_scales[l];
					}
					new_weights[l] = 0;
					new_shifts[l] = 0;
					new_scales[l] = 0;
				}
			}
		}

		cout << "thats all" << endl;
		for (t = 0; t < u_new_n; ++t) {
			delete[] new_g_ij[t];
			delete[] y_i_j[t];
			delete[] new_g_ij_0[t];
		}

		delete[] new_g_ij;
		delete[] y_i_j;
		delete[] new_g_ij_0;
		delete[] new_weights;
		delete[] new_shifts;
		delete[] new_scales;

	};

	auto begin1 = std::chrono::steady_clock::now();
	std::thread threadObj1(mix_part_computation, 0);
	std::thread threadObj2(mix_part_computation, 1);
	std::thread threadObj3(mix_part_computation, 2);
	std::thread threadObj4(mix_part_computation, 3);
	std::thread threadObj5(mix_part_computation, 4);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n";
	delete[] class_center;

}

// SEM with OPenMP

void mixture_handler::SEMalgorithm_opMP() {
	int h_w = (window_size / 2);
	int amount_window = img_l_x / h_w-1;
	re_comp_shifts = new long double*[amount_window*amount_window];
	re_comp_scales = new long double*[amount_window*amount_window];
	bool     stop_flag = true;
	int* iters = new int[amount_window*amount_window];
	int i_new_n = window_size * window_size;
	int i_n_add = img_l_x - (img_l_x / h_w)* h_w;
	unsigned i, t, r;

	for (i = 0; i < amount_window*amount_window; ++i) {
		re_comp_shifts[i] = new long double[hyp_cl_amount];
		re_comp_scales[i] = new long double[hyp_cl_amount];
		iters[i] = 0;
		for (t = 0; t < hyp_cl_amount; ++t) {
			re_comp_shifts[i][t] = 0;
			re_comp_scales[i][t] = 0;
		}
	}

	unsigned block_size = img_l_x / thr_nmb;
	
	/*unsigned vert = block_size / h_w;
	unsigned horiz = img_l / h_w - 1;
	amount_trg = vert * horiz;
	unsigned k = hyp_cl_amount;
	
	long double new_n = window_size * window_size;
	const unsigned u_new_n = window_size * window_size;*/
	
	
	double   max_buf = 0;
	auto begin1 = std::chrono::steady_clock::now();
	cout << "omp_get_num_threads();  " << omp_get_num_threads() << endl;
	#pragma omp parallel 
	{
		double   summ = 0;
		double   pix_buf = 0;
		double        cur_max = 0;
		int loc_amount_window = amount_window;
		int loc_hyp_cl_amount = hyp_cl_amount;
		int loc_window_size = window_size;
		double loc_accuracy = accuracy;
		//cout << "omp_get_num_threads();  " << omp_get_num_threads() << endl;
		double** new_g_ij = new double*[(loc_window_size+ i_n_add)*(loc_window_size + i_n_add)];
		double** new_g_ij_0 = new double*[(loc_window_size + i_n_add)*(loc_window_size + i_n_add)];
		double**    y_i_j = new double*[(loc_window_size + i_n_add)*(loc_window_size + i_n_add)];
		double * new_weights = new  double[loc_hyp_cl_amount];
		double * new_shifts = new  double[loc_hyp_cl_amount];
		double * new_scales = new  double[loc_hyp_cl_amount];
		bool          stop_flag = true;
		const double  sq_pi = sqrt(2 * pi);
		double        min_p = double(min_trg_size* min_trg_size) / double(loc_window_size * loc_window_size);
		double buf_max = 0;
		int      idx_max = 0;
		double val;
		
		int x_min, y_min,   i_lim, i_step, itr, horiz,vert, j, l, t;
		int u_new_n = 0;
		u_new_n = loc_window_size * loc_window_size;
		horiz = loc_window_size;
		vert = loc_window_size;
		double d_u_new_n = double(u_new_n);
		double bound_d, bound_u, psumm;
		boost::random::uniform_01 <> dist_poly;
		boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
		for (t = 0; t < loc_hyp_cl_amount; t++) {
			new_weights[t] = 0;
			new_shifts[t] = 0;
			new_scales[t] = 0;
		}
		for (l = 0; l < (loc_window_size + i_n_add)*(loc_window_size + i_n_add); ++l) {
			new_g_ij[l]   = new double[loc_hyp_cl_amount];
			new_g_ij_0[l] = new double[loc_hyp_cl_amount];
			y_i_j[l]      = new double[loc_hyp_cl_amount];
			psumm = 0;
			for (t = 0; t < loc_hyp_cl_amount; t++) {
					val = dist_poly(generator);
					psumm += val;
					new_g_ij[l][t] = val;
					new_g_ij_0[l][t] = 0;
					//y_i_j[l][t] = 0;
			}
			for ( t = 0; t < loc_hyp_cl_amount; t++)
				new_g_ij[l][t] = new_g_ij[l][t] / psumm;
		}
		
		
		#pragma omp for  
		for (int r = 0; r < loc_amount_window; ++r) {
		 
			x_min = r*h_w;
			for (j = 0; j < loc_amount_window; ++j) {
				y_min = j * h_w;
				stop_flag = true;
				//cur_max = 0;
				itr = 0;
				/*
					if (r != amount_window - 1)
						horiz = window_size;
					else
						horiz = window_size + i_n_add;
					if (j != amount_window - 1)
						vert = window_size;
					else
						vert = (window_size + i_n_add);
					u_new_n = horiz * vert;
				*/
				//# pragma omp critical (bound1)
				//cout << new_g_ij[0][0] << endl;
					while (stop_flag &&  itr < 300) {
						++itr;
						cur_max = 0;
						for (l = 0; l < u_new_n; ++l) {
							summ = 0;
							bound_d = 0;
							bound_u = new_g_ij[l][0];
							val = dist_poly(generator);
							for ( t = 0; t < loc_hyp_cl_amount; ++t) {
								if ((val < bound_u)
									&& (val >= bound_d)) {
									y_i_j[l][t] = 1;
									++new_weights[t] ;
									new_shifts[t] += my_picture[x_min + l / horiz][y_min + l % horiz];
								}
								else
									y_i_j[l][t] = 0;
								bound_d += new_g_ij[l][t];
								if (t < loc_hyp_cl_amount - 2)
									bound_u += new_g_ij[l][t + 1];
								else
									if (t == loc_hyp_cl_amount - 2)
										bound_u = 1;
							}
						}

						for ( t = 0; t < u_new_n; ++t) {
							pix_buf = my_picture[x_min + t / horiz][y_min + t % horiz];
							for (l = 0; l < loc_hyp_cl_amount; ++l) {
								if (new_weights[l] > 0) {
									
									
									if (t == 0)
										new_shifts[l] = new_shifts[l] / new_weights[l];
									new_scales[l] += y_i_j[t][l] * (pix_buf
										- new_shifts[l])*(pix_buf - new_shifts[l]);
									if (t == u_new_n - 1) {
										new_scales[l] = sqrt(new_scales[l] /new_weights[l]);
										new_weights[l] = new_weights[l] / d_u_new_n;
									}
								}
								
								if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
									cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
								new_g_ij_0[t][l] = new_g_ij[t][l];
							}
							
						}
						for (t = 0; t < u_new_n; ++t) {
							pix_buf = my_picture[x_min + t / horiz][y_min + t % horiz];
							
							summ = 0;

							for (l = 0; l < loc_hyp_cl_amount; ++l) {

								if ((new_weights[l] >= 0)
									&& (new_scales[l] >= 0))
									summ += new_weights[l] * (1 / (new_scales[l] * sq_pi))*exp(-((pix_buf
										- new_shifts[l])*(pix_buf - new_shifts[l])) / (2.0 * new_scales[l] * new_scales[l]));

							}
							for (l = 0; l < loc_hyp_cl_amount; ++l)
								if ((new_weights[l] >= 0)
									&& (new_scales[l] >= 0))
									new_g_ij[t][l] = new_weights[l] * (1 / (new_scales[l] * sq_pi*summ))*exp(-((pix_buf
										- new_shifts[l])*(pix_buf - new_shifts[l])) / (2.0 * new_scales[l] * new_scales[l]));
								else
									new_g_ij[t][l] = 0;
						}
						
						if (stop_flag) {
							if (cur_max < loc_accuracy)
								stop_flag = false;
							else {
								cur_max = 0;
								for ( l = 0; l < loc_hyp_cl_amount; ++l) {
									new_weights[l] = 0;
									new_shifts[l] = 0;
									new_scales[l] = 0;
								}
							}
						}
					}
					for ( l = 0; l < loc_hyp_cl_amount; ++l)
						new_weights[l] = 0;
					for ( t = 0; t < u_new_n; ++t) {
						
						buf_max = new_g_ij[t][0];
						idx_max = 0;
						psumm = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							if (buf_max < new_g_ij[t][l]) {
								buf_max = new_g_ij[t][l];
								idx_max = l;
							}
							new_g_ij_0[t][l] = 0;
							val = dist_poly(generator);
							psumm += val;
							new_g_ij[t][l] = val;
							new_g_ij_0[t][l] = 0;
						}
						++new_weights[idx_max];
						for ( l = 0; l < loc_hyp_cl_amount; ++l)
							new_g_ij[t][l] = new_g_ij[t][l] / psumm;
						//new_g_ij[t][l] = 1.0 / double(k);

					}
					for ( l = 0; l < loc_hyp_cl_amount; ++l) {
						//cout << new_shifts[l] << endl;
						if ((new_weights[l] / d_u_new_n > min_p)
							&& (new_scales[l] > 0)) {

							re_comp_shifts[ r *  loc_amount_window + j][l] = new_shifts[l];
							re_comp_scales[ r * loc_amount_window + j][l] = new_scales[l];
						}
						new_weights[l] = 0;
						new_shifts[l] = 0;
						new_scales[l] = 0;
					}
					
			}
			/* # pragma omp critical 
				cout << r<<"  "<<j << endl;*/
			
		}

		cout << "thats all" << endl;
		for (t = 0; t < (loc_window_size + i_n_add)*(loc_window_size + i_n_add); ++t) {
			delete[] new_g_ij[t];
			delete[] y_i_j[t];
			delete[] new_g_ij_0[t];
		}

		delete[] new_g_ij;
		delete[] y_i_j;
		delete[] new_g_ij_0;
		delete[] new_weights;
		delete[] new_shifts;
		delete[] new_scales;



	}

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n";
	
}

//2 vers openmp

void mixture_handler::SEMalgorithm_opMP2() {
	int h_w = (window_size / 2);
	int amount_window = img_l_x / h_w - 1;
	re_comp_shifts = new long double*[amount_window*amount_window];
	re_comp_scales = new long double*[amount_window*amount_window];
	bool     stop_flag = true;
	
	int i_new_n = window_size * window_size;
	int i_n_add = img_l_x - (img_l_x / h_w)* h_w;
	unsigned i, t, r;
	double        cur_max = 0;
	const double  sq_pi = sqrt(2 * pi);
	double        min_p = double(min_trg_size* min_trg_size) / double(window_size * window_size);

	int x_min, y_min, i_lim, i_step, itr, horiz, vert, j;
	int u_new_n = 0;
	u_new_n = window_size * window_size;
	horiz = window_size;
	vert = window_size;
	double d_u_new_n = double(u_new_n);

	for (i = 0; i < amount_window*amount_window; ++i) {
		re_comp_shifts[i] = new long double[hyp_cl_amount];
		re_comp_scales[i] = new long double[hyp_cl_amount];
		
		for (t = 0; t < hyp_cl_amount; ++t) {
			re_comp_shifts[i][t] = 0;
			re_comp_scales[i][t] = 0;
		}
	}

	
	double   max_buf = 0;
	auto begin1 = std::chrono::steady_clock::now();
	
	double** new_g_ij = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double** new_g_ij_0 = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double**    y_i_j = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double * new_weights = new  double[hyp_cl_amount];
	double * new_shifts  = new  double[hyp_cl_amount];
	double * new_scales  = new  double[hyp_cl_amount];
	for (int t = 0; t < hyp_cl_amount; t++) {
		new_weights[t] = 0;
		new_shifts[t] = 0;
		new_scales[t] = 0;
	}
	
	double val1, psumm1;
	boost::random::uniform_01 <> dist_poly1;
	boost::random::mt19937 generator1{ static_cast<std::uint32_t>(time(0)) };
	for (int l = 0; l < (window_size + i_n_add)*(window_size + i_n_add); ++l) {
		new_g_ij[l] = new double[hyp_cl_amount];
		new_g_ij_0[l] = new double[hyp_cl_amount];
		y_i_j[l] = new double[hyp_cl_amount];
		psumm1 = 0;
		for (int t = 0; t < hyp_cl_amount; t++) {
			val1 = dist_poly1(generator1);
			psumm1 += val1;
			new_g_ij[l][t] = val1;
			new_g_ij_0[l][t] = 0;
		}
		for (int t = 0; t < hyp_cl_amount; t++)
			new_g_ij[l][t] = new_g_ij[l][t] / psumm1;
	}

	for (int r = 0; r < amount_window; ++r) {
		x_min = r * h_w;
		for (j = 0; j < amount_window; ++j) {
			y_min = j * h_w;
			stop_flag = true;
			itr = 0;
			/*
				if (r != amount_window - 1)
					horiz = window_size;
				else
					horiz = window_size + i_n_add;
				if (j != amount_window - 1)
					vert = window_size;
				else
					vert = (window_size + i_n_add);
				u_new_n = horiz * vert;
			*/
				
			while (stop_flag &&  itr < 300) {
				++itr;
				cur_max = 0;
				#pragma omp parallel 
				{
					int loc_hyp_cl_amount = hyp_cl_amount;
					int u_new_n  = window_size * window_size;
						
					double bound_d, bound_u, psumm, val;
					boost::random::uniform_01 <> dist_poly;
					boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
					#pragma omp for  
					for (int l = 0; l < u_new_n; ++l) {
							
						bound_d = 0;
						bound_u = new_g_ij[l][0];
						val = dist_poly(generator);
						for (int t = 0; t < loc_hyp_cl_amount; ++t) {
							if ((val < bound_u)
								&& (val >= bound_d)) {
								y_i_j[l][t] = 1;
								# pragma omp critical
								{
									++new_weights[t];
									new_shifts[t] += my_picture[x_min + l / horiz][y_min + l % horiz];
								}

							}
							else
								y_i_j[l][t] = 0;
							bound_d += new_g_ij[l][t];
							if (t < loc_hyp_cl_amount - 2)
								bound_u += new_g_ij[l][t + 1];
							else
								if (t == loc_hyp_cl_amount - 2)
									bound_u = 1;
						}
					}
				}
				for (int l = 0; l < hyp_cl_amount; ++l) 
					new_shifts[l] = new_shifts[l] / new_weights[l];
				#pragma omp parallel 
				{
					double   pix_buf = 0;
					int l;
						
					int loc_hyp_cl_amount = hyp_cl_amount;
					int u_new_n = window_size * window_size;
					double d_u_new_n = double(u_new_n);
					#pragma omp for  
					for (int t = 0; t < u_new_n; ++t) {
						pix_buf = my_picture[x_min + t / horiz][y_min + t % horiz];
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							# pragma omp critical
							{
								if (new_weights[l] > 0) {
									new_scales[l] += y_i_j[t][l] * (pix_buf
										- new_shifts[l])*(pix_buf - new_shifts[l]);
								}

								if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
									cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
							}
							new_g_ij_0[t][l] = new_g_ij[t][l];
						}

					}
				}
				/*for (int t = 0; t < u_new_n; ++t) {
						
					for (int l = 0; l < hyp_cl_amount; ++l) {


							if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
								cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
							
						new_g_ij_0[t][l] = new_g_ij[t][l];
					}

				}*/
				for (int l = 0; l < hyp_cl_amount; ++l) {
					new_scales[l] = sqrt(new_scales[l] / new_weights[l]);
					new_weights[l] = new_weights[l] / d_u_new_n;
				}
				#pragma omp parallel 
				{
					double   pix_buf = 0;
					double summ;
					int l;
						
					int loc_hyp_cl_amount = hyp_cl_amount;
					int u_new_n = window_size * window_size;
					double d_u_new_n = double(u_new_n);
					#pragma omp for  
					for (int t = 0; t < u_new_n; ++t) {
						pix_buf = my_picture[x_min + t / horiz][y_min + t % horiz];
						summ = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							if ((new_weights[l] >= 0)
								&& (new_scales[l] > 0))
								summ += new_weights[l] * (1 / (new_scales[l] * sq_pi))*exp(-((pix_buf
									- new_shifts[l])*(pix_buf - new_shifts[l])) / (2.0 * new_scales[l] * new_scales[l]));

						}
						for (l = 0; l < loc_hyp_cl_amount; ++l)
							if ((new_weights[l] >= 0)
								&& (new_scales[l] > 0))
								new_g_ij[t][l] = new_weights[l] * (1 / (new_scales[l] * sq_pi*summ))*exp(-((pix_buf
									- new_shifts[l])*(pix_buf - new_shifts[l])) / (2.0 * new_scales[l] * new_scales[l]));
							else
								new_g_ij[t][l] = 0;

					}
				}

				if (stop_flag) {
					if (cur_max < accuracy)
						stop_flag = false;
					else {
						cur_max = 0;
						for (int l = 0; l < hyp_cl_amount; ++l) {
							new_weights[l] = 0;
							new_shifts[l] = 0;
							new_scales[l] = 0;
						}
					}
				}
			}
			for (int l = 0; l < hyp_cl_amount; ++l)
				new_weights[l] = 0;
			#pragma omp parallel 
			{
				double buf_max = 0;
				int      idx_max = 0;
				double val, psumm;
				int l;
				int loc_amount_window = amount_window;
				int loc_hyp_cl_amount = hyp_cl_amount;
				int loc_window_size = window_size;
					
				boost::random::uniform_01 <> dist_poly;
				boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
				#pragma omp for
				for (int t = 0; t < u_new_n; ++t) {

					buf_max = new_g_ij[t][0];
					idx_max = 0;
					psumm = 0;
					for (l = 0; l < loc_hyp_cl_amount; ++l) {
						if (buf_max < new_g_ij[t][l]) {
							buf_max = new_g_ij[t][l];
							idx_max = l;
						}
						new_g_ij_0[t][l] = 0;
						val = dist_poly(generator);
						psumm += val;
						new_g_ij[t][l] = val;
						new_g_ij_0[t][l] = 0;
					}
					#pragma omp critical
					++new_weights[idx_max];
					for (l = 0; l < loc_hyp_cl_amount; ++l)
						new_g_ij[t][l] = new_g_ij[t][l] / psumm;
					//new_g_ij[t][l] = 1.0 / double(k);

				}
			}
			//for (i = 0; i < img_l*img_l; ++i) {
			//	idx_i = i / img_l;
			//	idx_j = i % img_l;
			//	summ = 0;
			//	pix_buf = my_picture[idx_i][idx_j];
			//	for (j = 0; j < hyp_cl_amount; ++j)
			//		summ += mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)))*exp(-((pix_buf
			//			- mix_shift[j])*(pix_buf - mix_shift[j])) / (2.0 * mix_scale[j] * mix_scale[j]));

			//	big_summ += log(summ);
			//}
			//unsigned count_n_z = 0;
			///*for (j = 0; j < hyp_cl_amount; ++j)
			//	if (mix_weight[j] != 0)
			//		++count_n_z;*/
			//count_n_z = hyp_cl_amount;
			//bic_value = -2 * big_summ + log(img_l*img_l)*(3 * count_n_z - 1);
			for (int l = 0; l < hyp_cl_amount; ++l) {
				//cout << new_shifts[l] << endl;
				if ((new_weights[l] / d_u_new_n > min_p)
					&& (new_scales[l] > 0)) {
					
					re_comp_shifts[r *  amount_window + j][l] = new_shifts[l];
					re_comp_scales[r * amount_window + j][l] = new_scales[l];
				}
				new_weights[l] = 0;
				new_shifts[l] = 0;
				new_scales[l] = 0;
			}

		}
	}

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n";
	for (int t = 0; t < (window_size + i_n_add)*(window_size + i_n_add); ++t) {
		delete[] new_g_ij[t];
		delete[] y_i_j[t];
		delete[] new_g_ij_0[t];
	}

	delete[] new_g_ij;
	delete[] y_i_j;
	delete[] new_g_ij_0;
	delete[] new_weights;
	delete[] new_shifts;
	delete[] new_scales;
}

//sem заточенный под нормальное рапределение с распараллеливанием 
// если алгоритм не бить, проседает производительность

void mixture_handler::SEMalgorithm_opMP_normal_spec() {
	int h_w = (window_size / 2);
	int amount_window = img_l_x / h_w - 1;
	re_comp_shifts = new long double*[amount_window*amount_window];
	re_comp_scales = new long double*[amount_window*amount_window];
	bool     stop_flag = true;

	int i_new_n = window_size * window_size;
	int i_n_add = img_l_x - (img_l_x / h_w)* h_w;
	int est_amount = 2;
	unsigned i, t, r;
	double        cur_max = 0;
	const double  sq_pi = sqrt(2 * pi);
	double        min_p = double(min_trg_size* min_trg_size) / double(window_size * window_size);

	int x_min, y_min, i_lim, i_step, itr, horiz, vert, j;
	int u_new_n = 0;
	u_new_n = window_size * window_size;
	horiz = window_size;
	vert = window_size;
	double d_u_new_n = double(u_new_n);

	for (i = 0; i < amount_window*amount_window; ++i) {
		re_comp_shifts[i] = new long double[hyp_cl_amount];
		re_comp_scales[i] = new long double[hyp_cl_amount];
		for (t = 0; t < hyp_cl_amount; ++t) {
			re_comp_shifts[i][t] = 0;
			re_comp_scales[i][t] = 0;
		}
	}


	double   max_buf = 0;
	double bic_value = 0;
	double pre_bic_value = 0;
	int pre_amount = 1;
	auto begin1 = std::chrono::steady_clock::now();

	double** new_g_ij = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double** new_g_ij_0 = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double**    y_i_j = new double*[(window_size + i_n_add)*(window_size + i_n_add)];

	double * new_weights_mean = new  double[hyp_cl_amount];

	double ** new_shifts = new double*[est_amount];
	double ** new_scales = new double*[est_amount];

	for (j = 0; j < est_amount; ++j) {
		new_shifts[j] = new  double[hyp_cl_amount];
		new_scales[j] = new  double[hyp_cl_amount];
	}
	

	double** new_med_sample = new double*[hyp_cl_amount];

	double * pre_new_weights = new double[hyp_cl_amount];
	double * pre_new_shifts  = new double[hyp_cl_amount];
	double * pre_new_scales  = new double[hyp_cl_amount];

	double * max_L_mass = new  double[est_amount];

	for (j = 0; j < hyp_cl_amount; ++j) {
		pre_new_weights[j] = 0;
		pre_new_shifts[j]  = 0;
		pre_new_scales[j]  = 0;
		new_med_sample[j] = new double[(window_size + i_n_add)*(window_size + i_n_add)];

		new_weights_mean[j] = 0;

		for (int l = 0; l < est_amount; l++) {
			new_shifts[l][j] = 0;
			new_scales[l][j] = 0;
			max_L_mass[l] = 0;
		}
	}

	double val1, psumm1;
	boost::random::uniform_01 <> dist_poly1;
	boost::random::mt19937 generator1{ static_cast<std::uint32_t>(time(0)) };
	for (j = 0; j < (window_size + i_n_add)*(window_size + i_n_add); ++j) {
		new_g_ij[j]   = new double[hyp_cl_amount];
		new_g_ij_0[j] = new double[hyp_cl_amount];
		y_i_j[j]      = new double[hyp_cl_amount];
	}
	bool* med_flag = new bool[est_amount];
	med_flag[0] = true;
	for (int r = 0; r < amount_window; ++r) {
		x_min = r * h_w;
		for (j = 0; j < amount_window; ++j) {
			y_min = j * h_w;
			for (int itr_cl_am = 1; itr_cl_am <= hyp_cl_amount; ++itr_cl_am) {
				stop_flag = true;
				itr = 0;
				/*for (int l = 0; l < (window_size + i_n_add)*(window_size + i_n_add); ++l) {
					psumm1 = 0;
					for (int t = 0; t < itr_cl_am; t++) {
						val1 = dist_poly1(generator1);
						psumm1 += val1;
						new_g_ij[l][t] = val1;
						new_g_ij_0[l][t] = 0;
					}
					for (int t = 0; t < itr_cl_am; t++)
						new_g_ij[l][t] = new_g_ij[l][t] / psumm1;
				}
*/
				#pragma omp parallel 
				{
					double val, psumm;
					int l;
					int u_new_n = window_size * window_size;
					int loc_hyp_cl_amount = itr_cl_am;
					boost::random::uniform_01 <> dist_poly;
					boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
					#pragma omp for
					for (int t = 0; t < u_new_n; ++t) {
						psumm = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							val = dist_poly(generator);
							psumm += val;
							new_g_ij[t][l] = val;
							new_g_ij_0[t][l] = 0;
						}
					
						for (l = 0; l < loc_hyp_cl_amount; ++l)
							new_g_ij[t][l] = new_g_ij[t][l] / psumm;
					}
				}
				/*
					if (r != amount_window - 1)
						horiz = window_size;
					else
						horiz = window_size + i_n_add;
					if (j != amount_window - 1)
						vert = window_size;
					else
						vert = (window_size + i_n_add);
					u_new_n = horiz * vert;
				*/
				
				while (stop_flag &&  itr < 300) {
					++itr;
					cur_max = 0;
                    #pragma omp parallel 
					{
						int loc_hyp_cl_amount = itr_cl_am;
						int u_new_n = window_size * window_size;
						int t;
						int loc_horiz = horiz;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						double bound_d, bound_u, psumm, val, pix_buf;
						boost::random::uniform_01 <> dist_poly;
						boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
                        #pragma omp for  
						for (int l = 0; l < u_new_n; ++l) {
							pix_buf = my_picture[loc_x_min + l / loc_horiz][loc_y_min + l % loc_horiz];
							bound_d = 0;
							bound_u = new_g_ij[l][0];
							val = dist_poly(generator);
							for (t = 0; t < loc_hyp_cl_amount; ++t) {
								if ((val < bound_u)
									&& (val >= bound_d)) {
									y_i_j[l][t] = 1;
									# pragma omp critical
									{
										new_med_sample[t][int(new_weights_mean[t])] = pix_buf;
										++new_weights_mean[t];
										new_shifts[0][t] += pix_buf;
									}
								}
								else
									y_i_j[l][t] = 0;
								bound_d += new_g_ij[l][t];
								if (t < loc_hyp_cl_amount - 2)
									bound_u += new_g_ij[l][t + 1];
								else
									if (t == loc_hyp_cl_amount - 2)
										bound_u = 1;
							}
						}
					}
					
					
					med_flag[1] = true;
					for (int l = 0; l < itr_cl_am; ++l) {
						new_shifts[0][l] = new_shifts[0][l] / new_weights_mean[l];
						new_shifts[1][l] = find_med(new_med_sample[l], new_weights_mean[l]);
						if (new_shifts[1][l] < 0)
							med_flag[1] = false;
					}
					#pragma omp parallel 
					{
						string mixture_type = mixture_type;
						double   pix_buf = 0; int l;
						int loc_horiz = horiz;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						int loc_hyp_cl_amount = itr_cl_am;
						int u_new_n = window_size * window_size;
						double d_u_new_n = double(u_new_n);
						
						#pragma omp for  
						for (int t = 0; t < u_new_n; ++t) {
							pix_buf = my_picture[loc_x_min + t / loc_horiz][loc_y_min + t % loc_horiz];
							for (l = 0; l < loc_hyp_cl_amount; ++l) {
								
								# pragma omp critical
								{
									
									if (new_weights_mean[l] > 0) {
										new_scales[0][l] += y_i_j[t][l] * (pix_buf
											- new_shifts[0][l])*(pix_buf - new_shifts[0][l]);
										if (med_flag[1])
											new_scales[1][l] += y_i_j[t][l] * abs(pix_buf - new_shifts[1][l]);
									}
									
									if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
										cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
								}
								new_g_ij_0[t][l] = new_g_ij[t][l];
							}
						}
					}
					
					
					for (int l = 0; l < itr_cl_am; ++l) {
						
						new_scales[0][l] = sqrt(new_scales[0][l] / new_weights_mean[l]);
						new_scales[1][l] = new_scales[1][l] * 1.2533 / new_weights_mean[l];
						
						new_weights_mean[l] = new_weights_mean[l] / d_u_new_n;
					}
					#pragma omp parallel
					{

						double B, pix_buf;
						string mixture_type = mixture_type;
						bool pre_flag = true;
						bool flag = false;
						int l, t;
						int u_new_n = window_size * window_size;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						int loc_est_amount = est_amount;
						int loc_horiz = horiz;
						int loc_itr_cl_am = itr_cl_am;
						double d_u_new_n = double(u_new_n);
						#pragma omp for
						for (int m = 0; m < u_new_n; m++) {
							pix_buf = my_picture[loc_x_min + m / loc_horiz][loc_y_min + m % loc_horiz];
							flag = false;
							for (l = 0; l < loc_est_amount; ++l) {
								B = 0;
								for (t = 0; t < loc_itr_cl_am; ++t) {
									if (((new_scales[l][t] != 0) && (new_weights_mean[t] != 0)) && (med_flag[l])) {
											B += new_weights_mean[t] * (1.0 / (new_scales[l][t]))*
												exp(-(pow(pix_buf - new_shifts[l][t], 2)) /
												(2.0 * new_scales[l][t] * new_scales[l][t]));
									}
									else {
										flag = true;
										break;
									}
								}
								#pragma omp critical
								if (B > 0)
									max_L_mass[l] += log(B / (d_u_new_n));
								else
									max_L_mass[l] += 0;
							}
						}
						
					}

					int max_idx = 0;
					
					for (int m = 0; m < est_amount; ++m) {
						//cout << m << " " << max_L_mass[m] << endl;
						if ((max_L_mass[max_idx] < max_L_mass[m])&&((med_flag[m])))
							max_idx = m;
					}

					for (int m = 0; m < itr_cl_am; ++m) {
						new_shifts[0][m] = new_shifts[max_idx][m];
						new_scales[0][m] = new_scales[max_idx][m];
					}
					/*cout << "max_idx " << max_idx<<" "<< new_shifts[1][0]<<" "<< new_scales[1][0]<< " " << new_shifts[1][1] << " " << new_scales[1][1] << " "<< med_flag[1]<<endl;
					cout << "max_idx_t "  << " " << new_shifts[0][0] << " " << new_scales[0][0] << " " << new_shifts[0][1] << " " << new_scales[0][1] << " " << med_flag[1] << endl;*/
					#pragma omp parallel 
					{
						double   pix_buf = 0;
						double summ;
						int l;
						string mixture_type = mixture_type;
						int loc_horiz = horiz;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						int loc_hyp_cl_amount = itr_cl_am;
						int u_new_n = window_size * window_size;
						double d_u_new_n = double(u_new_n);
						#pragma omp for  
						for (int t = 0; t < u_new_n; ++t) {
							pix_buf = my_picture[loc_x_min + t / loc_horiz][loc_y_min + t % loc_horiz];
							summ = 0;
							for (l = 0; l < loc_hyp_cl_amount; ++l) {
								if ((new_weights_mean[l] >= 0)
									&& (new_scales[0][l] > 0)) {
										summ += new_weights_mean[l] * (1 / (new_scales[0][l] * sq_pi))*exp(-((pix_buf
											- new_shifts[0][l])*(pix_buf - new_shifts[0][l])) / (2.0 * new_scales[0][l] * new_scales[0][l]));
								}

							}
							for (l = 0; l < loc_hyp_cl_amount; ++l)
								if ((new_weights_mean[l] >= 0)
									&& (new_scales[0][l] > 0))
										new_g_ij[t][l] = new_weights_mean[l] * (1 / (new_scales[0][l] * sq_pi*summ))*exp(-((pix_buf
											- new_shifts[0][l])*(pix_buf - new_shifts[0][l])) / (2.0 * new_scales[0][l] * new_scales[0][l]));
								else
									new_g_ij[t][l] = 0;

						}
					}

					if (stop_flag) {
						if (cur_max < accuracy)
							stop_flag = false;
						else {
							cur_max = 0;
							for (int l = 0; l < itr_cl_am; ++l) {
								new_weights_mean[l] = 0;
								for (int t = 0; t < est_amount; ++t) {
									new_shifts[t][l] = 0;
									new_scales[t][l] = 0;
									max_L_mass[t] = 0;
								}
							}
						}
					}
					
				}
				
				for (int l = 0; l < itr_cl_am; ++l) {
					new_weights_mean[l] = 0;
					//cout<<"dd "<< new_shifts[0][l]<< " "<<new_scales[0][l] <<endl;
				}
				#pragma omp parallel 
				{
					double buf_max = 0;
					int      idx_max = 0;
					double val, psumm;
					int l;
					int loc_hyp_cl_amount = itr_cl_am;
					boost::random::uniform_01 <> dist_poly;
					int u_new_n = window_size * window_size;
					boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
					#pragma omp for
					for (int t = 0; t < u_new_n; ++t) {

						buf_max = new_g_ij[t][0];
						idx_max = 0;
						psumm = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							if (buf_max < new_g_ij[t][l]) {
								buf_max = new_g_ij[t][l];
								idx_max = l;
							}
							new_g_ij_0[t][l] = 0;
							val = dist_poly(generator);
							psumm += val;
							new_g_ij[t][l] = val;
							new_g_ij_0[t][l] = 0;
						}
						#pragma omp critical
						++new_weights_mean[idx_max];
						for (l = 0; l < loc_hyp_cl_amount; ++l)
							new_g_ij[t][l] = new_g_ij[t][l] / psumm;
						//new_g_ij[t][l] = 1.0 / double(k);

					}
				}
				double big_summ = 0;
				#pragma omp parallel
				{
					
					double pix_buf, summ;
					int t; int loc_horiz = horiz;
					int loc_x_min = x_min;
					int loc_y_min = y_min;
					#pragma omp for
					for (int l = 0; l < u_new_n; ++l) {
						summ = 0;
						pix_buf = my_picture[loc_x_min + l / loc_horiz][loc_y_min + l % loc_horiz];
						for (t = 0; t < itr_cl_am; ++t)
							summ += new_weights_mean[t] * (1 / (new_scales[0][t] * sqrt(2 * pi)))*exp(-((pix_buf
									- new_shifts[0][t])*(pix_buf - new_shifts[0][t])) / (2.0 * new_scales[0][t] * new_scales[0][t]));
						# pragma omp critical
						big_summ += log(summ);
					}

				}
				bic_value = 2 * big_summ - log(u_new_n)*(3 * itr_cl_am - 1);

				if (itr_cl_am == 1) {
					pre_bic_value = bic_value;
					pre_amount = 1;
					for (int l = 0; l < itr_cl_am; ++l) {
						pre_new_weights[l] = new_weights_mean[l];
						pre_new_shifts[l] = new_shifts[0][l];
						pre_new_scales[l] = new_scales[0][l];
					}
				}
				else {
					if (bic_value > pre_bic_value) {
						pre_bic_value = bic_value;
						pre_amount = itr_cl_am;
						for (int l = 0; l < itr_cl_am; ++l) {
							pre_new_weights[l] = new_weights_mean[l] ;
							pre_new_shifts[l] = new_shifts[0][l];
							pre_new_scales[l] = new_scales[0][l];
						}
					}
				}
				for (int l = 0; l < itr_cl_am; ++l) {
					
					new_weights_mean[l] = 0;
					for (int t = 0; t < est_amount; ++t) {
						new_shifts[t][l] = 0;
						new_scales[t][l] = 0;
					}
				}
				
			}
			//cout << r << " " << j << " " << pre_amount<<endl;
			for (int l = 0; l < pre_amount; ++l) {
				//if(pre_amount==2)
				//cout << pre_new_shifts[l] << endl;
				if ((pre_new_weights[l] / d_u_new_n > min_p)
					&& (pre_new_scales[l] > 0)) {

					re_comp_shifts[r *  amount_window + j][l] = pre_new_shifts[l];
					re_comp_scales[r * amount_window + j][l] = pre_new_scales[l];
				}
				pre_new_weights[l] = 0;
				pre_new_shifts[l] = 0;
				pre_new_scales[l] = 0;
			}
		}
	}

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n";
	for (int t = 0; t < (window_size + i_n_add)*(window_size + i_n_add); ++t) {
		delete[] new_g_ij[t];
		delete[] y_i_j[t];
		delete[] new_g_ij_0[t];
	}

	delete[] new_g_ij;
	delete[] y_i_j;
	delete[] new_g_ij_0;
	delete[] med_flag;
	delete[] new_weights_mean;
	delete[] pre_new_weights;
	delete[] pre_new_shifts;
	delete[] pre_new_scales;
	delete[] max_L_mass;

	
	for (j = 0; j < est_amount; ++j) {
		delete[] new_shifts[j];
		delete[] new_scales[j];
	}
	delete[] new_shifts;
	delete[] new_scales;

	for (j = 0; j < hyp_cl_amount; ++j) {
		delete[] new_med_sample[j];
		
	}
	delete[] new_med_sample;
	

}

//

void mixture_handler::SEMalgorithm_opMP_rayleigh_spec() {
	int h_w = (window_size / 2);
	int amount_window = img_l_x / h_w - 1;
	re_comp_shifts = new long double*[amount_window*amount_window];
	re_comp_scales = new long double*[amount_window*amount_window];
	bool     stop_flag = true;

	int i_new_n = window_size * window_size;
	int i_n_add = img_l_x - (img_l_x / h_w)* h_w;
	int est_amount = 2;
	unsigned i, t, r;
	double        cur_max = 0;
	const double  sq_pi = sqrt(2 * pi);
	double        min_p = double(min_trg_size* min_trg_size) / double(window_size * window_size);

	int x_min, y_min, i_lim, i_step, itr, horiz, vert, j;
	int u_new_n = 0;
	u_new_n = window_size * window_size;
	horiz = window_size;
	vert = window_size;
	double d_u_new_n = double(u_new_n);

	for (i = 0; i < amount_window*amount_window; ++i) {
		re_comp_shifts[i] = new long double[hyp_cl_amount];
		re_comp_scales[i] = new long double[hyp_cl_amount];
		for (t = 0; t < hyp_cl_amount; ++t) {
			re_comp_shifts[i][t] = 0;
			re_comp_scales[i][t] = 0;
		}
	}


	double   max_buf = 0;
	double bic_value = 0;
	double pre_bic_value = 0;
	int pre_amount = 1;
	auto begin1 = std::chrono::steady_clock::now();

	double** new_g_ij = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double** new_g_ij_0 = new double*[(window_size + i_n_add)*(window_size + i_n_add)];
	double**    y_i_j = new double*[(window_size + i_n_add)*(window_size + i_n_add)];

	double * new_weights_mean = new  double[hyp_cl_amount];

	double ** new_shifts = new double*[est_amount];
	double ** new_scales = new double*[est_amount];

	for (j = 0; j < est_amount; ++j) {
		new_shifts[j] = new  double[hyp_cl_amount];
		new_scales[j] = new  double[hyp_cl_amount];
	}


	double** new_med_sample = new double*[hyp_cl_amount];

	double * pre_new_weights = new double[hyp_cl_amount];
	double * pre_new_shifts = new double[hyp_cl_amount];
	double * pre_new_scales = new double[hyp_cl_amount];

	double * max_L_mass = new  double[est_amount];

	for (j = 0; j < hyp_cl_amount; ++j) {
		pre_new_weights[j] = 0;
		pre_new_shifts[j] = 0;
		pre_new_scales[j] = 0;
		new_med_sample[j] = new double[(window_size + i_n_add)*(window_size + i_n_add)];

		new_weights_mean[j] = 0;

		for (int l = 0; l < est_amount; l++) {
			new_shifts[l][j] = 0;
			new_scales[l][j] = 0;
			max_L_mass[l] = 0;
		}
	}

	double val1, psumm1;
	boost::random::uniform_01 <> dist_poly1;
	boost::random::mt19937 generator1{ static_cast<std::uint32_t>(time(0)) };
	for (j = 0; j < (window_size + i_n_add)*(window_size + i_n_add); ++j) {
		new_g_ij[j] = new double[hyp_cl_amount];
		new_g_ij_0[j] = new double[hyp_cl_amount];
		y_i_j[j] = new double[hyp_cl_amount];
	}
	bool* med_flag = new bool[est_amount];
	med_flag[0] = true;
	for (int r = 0; r < amount_window; ++r) {
		x_min = r * h_w;
		for (j = 0; j < amount_window; ++j) {
			y_min = j * h_w;
			for (int itr_cl_am = 1; itr_cl_am <= hyp_cl_amount; ++itr_cl_am) {
				stop_flag = true;
				itr = 0;
				/*for (int l = 0; l < (window_size + i_n_add)*(window_size + i_n_add); ++l) {
					psumm1 = 0;
					for (int t = 0; t < itr_cl_am; t++) {
						val1 = dist_poly1(generator1);
						psumm1 += val1;
						new_g_ij[l][t] = val1;
						new_g_ij_0[l][t] = 0;
					}
					for (int t = 0; t < itr_cl_am; t++)
						new_g_ij[l][t] = new_g_ij[l][t] / psumm1;
				}
*/
#pragma omp parallel 
				{
					double val, psumm;
					int l;
					int u_new_n = window_size * window_size;
					int loc_hyp_cl_amount = itr_cl_am;
					boost::random::uniform_01 <> dist_poly;
					boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
#pragma omp for
					for (int t = 0; t < u_new_n; ++t) {
						psumm = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							val = dist_poly(generator);
							psumm += val;
							new_g_ij[t][l] = val;
							new_g_ij_0[t][l] = 0;
						}

						for (l = 0; l < loc_hyp_cl_amount; ++l)
							new_g_ij[t][l] = new_g_ij[t][l] / psumm;
					}
				}
				/*
					if (r != amount_window - 1)
						horiz = window_size;
					else
						horiz = window_size + i_n_add;
					if (j != amount_window - 1)
						vert = window_size;
					else
						vert = (window_size + i_n_add);
					u_new_n = horiz * vert;
				*/

				while (stop_flag &&  itr < 300) {
					++itr;
					cur_max = 0;
#pragma omp parallel 
					{
						int loc_hyp_cl_amount = itr_cl_am;
						string mixture_type = mixture_type;
						int u_new_n = window_size * window_size;
						int t;
						int loc_horiz = horiz;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						double bound_d, bound_u, psumm, val, pix_buf;
						boost::random::uniform_01 <> dist_poly;
						boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
#pragma omp for  
						for (int l = 0; l < u_new_n; ++l) {
							pix_buf = my_picture[loc_x_min + l / loc_horiz][loc_y_min + l % loc_horiz];
							bound_d = 0;
							bound_u = new_g_ij[l][0];
							val = dist_poly(generator);
							for (t = 0; t < loc_hyp_cl_amount; ++t) {
								if ((val < bound_u)
									&& (val >= bound_d)) {
									y_i_j[l][t] = 1;
# pragma omp critical
									{
										new_med_sample[t][int(new_weights_mean[t])] = pix_buf;
										++new_weights_mean[t];
										new_scales[0][t] += pix_buf * pix_buf;
										
									}

								}
								else
									y_i_j[l][t] = 0;
								bound_d += new_g_ij[l][t];
								if (t < loc_hyp_cl_amount - 2)
									bound_u += new_g_ij[l][t + 1];
								else
									if (t == loc_hyp_cl_amount - 2)
										bound_u = 1;
							}
						}
					}


					med_flag[1] = true;
					for (int l = 0; l < itr_cl_am; ++l) {
							new_shifts[0][l] = 0;
							new_shifts[1][l] = 0;
							new_scales[0][l] = sqrt(new_scales[0][l] / double(2 * new_weights_mean[l]));

							new_scales[1][l] = find_med(new_med_sample[l], new_weights_mean[l]) / sqrt(log(4.0));
							if (new_scales[1][l] < 0)
								med_flag[1] = false;
					}
#pragma omp parallel 
					{
						
						double   pix_buf = 0; int l;
						int loc_horiz = horiz;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						int loc_hyp_cl_amount = itr_cl_am;
						int u_new_n = window_size * window_size;
						double d_u_new_n = double(u_new_n);

#pragma omp for  
						for (int t = 0; t < u_new_n; ++t) {
							pix_buf = my_picture[loc_x_min + t / loc_horiz][loc_y_min + t % loc_horiz];
							for (l = 0; l < loc_hyp_cl_amount; ++l) {

# pragma omp critical
								{
									
									if (cur_max < abs(new_g_ij[t][l] - new_g_ij_0[t][l]))
										cur_max = abs(new_g_ij[t][l] - new_g_ij_0[t][l]);
								}
								new_g_ij_0[t][l] = new_g_ij[t][l];
							}
						}
					}


					for (int l = 0; l < itr_cl_am; ++l) {
						
						new_weights_mean[l] = new_weights_mean[l] / d_u_new_n;
					}
#pragma omp parallel
					{

						double B, pix_buf;
						
						bool pre_flag = true;
						bool flag = false;
						int l, t;
						int u_new_n = window_size * window_size;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						int loc_est_amount = est_amount;
						int loc_horiz = horiz;
						int loc_itr_cl_am = itr_cl_am;
						double d_u_new_n = double(u_new_n);
#pragma omp for
						for (int m = 0; m < u_new_n; m++) {
							pix_buf = my_picture[loc_x_min + m / loc_horiz][loc_y_min + m % loc_horiz];
							flag = false;
							for (l = 0; l < loc_est_amount; ++l) {
								B = 0;
								for (t = 0; t < loc_itr_cl_am; ++t) {
									if (((new_scales[l][t] != 0) && (new_weights_mean[t] != 0)) && (med_flag[l])) {
											B += new_weights_mean[t] * (pix_buf / (new_scales[l][t] * new_scales[l][t]))*
											exp(-(pow(pix_buf, 2)) /
											(2.0 * new_scales[l][t] * new_scales[l][t]));
									}
									else {
										flag = true;
										break;
									}
								}
#pragma omp critical
								if (B > 0)
									max_L_mass[l] += log(B / (d_u_new_n));
								else
									max_L_mass[l] += 0;
							}
						}

					}

					int max_idx = 0;

					for (int m = 0; m < est_amount; ++m) {
						//cout << m << " " << max_L_mass[m] << endl;
						if ((max_L_mass[max_idx] < max_L_mass[m]) && ((med_flag[m])))
							max_idx = m;
					}

					for (int m = 0; m < itr_cl_am; ++m) {
						new_shifts[0][m] = new_shifts[max_idx][m];
						new_scales[0][m] = new_scales[max_idx][m];
					}
					//cout << "max_idx " << max_idx<<" "<< new_shifts[1][0]<<" "<< new_scales[1][0]<< " " << new_shifts[1][1] << " " << new_scales[1][1] << " "<< med_flag[1]<<endl;
					//cout << "max_idx_t "  << " " << new_shifts[0][0] << " " << new_scales[0][0] << " " << new_shifts[0][1] << " " << new_scales[0][1] << " " << med_flag[1] << endl;
#pragma omp parallel 
					{
						double   pix_buf = 0;
						double summ;
						int l;
					
						int loc_horiz = horiz;
						int loc_x_min = x_min;
						int loc_y_min = y_min;
						int loc_hyp_cl_amount = itr_cl_am;
						int u_new_n = window_size * window_size;
						double d_u_new_n = double(u_new_n);
#pragma omp for  
						for (int t = 0; t < u_new_n; ++t) {
							pix_buf = my_picture[loc_x_min + t / loc_horiz][loc_y_min + t % loc_horiz];
							summ = 0;
							for (l = 0; l < loc_hyp_cl_amount; ++l) {
								if ((new_weights_mean[l] >= 0)
									&& (new_scales[0][l] > 0)) {
									
											summ += new_weights_mean[l] * (pix_buf / (new_scales[0][l] * new_scales[0][l]))*exp(-((pix_buf
												)*(pix_buf)) / (2.0 * new_scales[0][l] * new_scales[0][l]));
									
								}

							}
							for (l = 0; l < loc_hyp_cl_amount; ++l)
								if ((new_weights_mean[l] >= 0)
									&& (new_scales[0][l] > 0))
									
											new_g_ij[t][l] = new_weights_mean[l] * (pix_buf / (new_scales[0][l] * new_scales[0][l]* summ))*exp(-(pix_buf
												*pix_buf) / (2.0 * new_scales[0][l] * new_scales[0][l]));
									
								else
									new_g_ij[t][l] = 0;

						}
					}

					if (stop_flag) {
						if (cur_max < accuracy)
							stop_flag = false;
						else {
							cur_max = 0;
							for (int l = 0; l < itr_cl_am; ++l) {
								new_weights_mean[l] = 0;
								for (int t = 0; t < est_amount; ++t) {
									new_shifts[t][l] = 0;
									new_scales[t][l] = 0;
									max_L_mass[t] = 0;
								}
							}
						}
					}

				}

				for (int l = 0; l < itr_cl_am; ++l) {
					new_weights_mean[l] = 0;
					//cout<<"dd "<< new_shifts[0][l]<< " "<<new_scales[0][l] <<endl;
				}
#pragma omp parallel 
				{
					double buf_max = 0;
					int      idx_max = 0;
					double val, psumm;
					int l;
					int loc_hyp_cl_amount = itr_cl_am;
					boost::random::uniform_01 <> dist_poly;
					int u_new_n = window_size * window_size;
					boost::random::mt19937 generator{ static_cast<std::uint32_t>(time(0)) };
#pragma omp for
					for (int t = 0; t < u_new_n; ++t) {

						buf_max = new_g_ij[t][0];
						idx_max = 0;
						psumm = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							if (buf_max < new_g_ij[t][l]) {
								buf_max = new_g_ij[t][l];
								idx_max = l;
							}
							new_g_ij_0[t][l] = 0;
							val = dist_poly(generator);
							psumm += val;
							new_g_ij[t][l] = val;
							new_g_ij_0[t][l] = 0;
						}
#pragma omp critical
						++new_weights_mean[idx_max];
						for (l = 0; l < loc_hyp_cl_amount; ++l)
							new_g_ij[t][l] = new_g_ij[t][l] / psumm;
						//new_g_ij[t][l] = 1.0 / double(k);

					}
				}
				double big_summ = 0;
#pragma omp parallel
				{
					string mixture_type = mixture_type;
					double pix_buf, summ;
					int t; int loc_horiz = horiz;
					int loc_x_min = x_min;
					int loc_y_min = y_min;
#pragma omp for
					for (int l = 0; l < u_new_n; ++l) {
						summ = 0;
						pix_buf = my_picture[loc_x_min + l / loc_horiz][loc_y_min + l % loc_horiz];
						for (t = 0; t < itr_cl_am; ++t)
							
									summ += new_weights_mean[t] * (pix_buf / (new_scales[0][t] * new_scales[0][t]))*exp(-((pix_buf
										)*(pix_buf)) / (2.0 * new_scales[0][t] * new_scales[0][t]));
							
# pragma omp critical
						big_summ += log(summ);
					}

				}
				bic_value = 2 * big_summ - log(u_new_n)*(2 * itr_cl_am - 1);

				if (itr_cl_am == 1) {
					pre_bic_value = bic_value;
					pre_amount = 1;
					for (int l = 0; l < itr_cl_am; ++l) {
						pre_new_weights[l] = new_weights_mean[l];
						pre_new_shifts[l] = new_shifts[0][l];
						pre_new_scales[l] = new_scales[0][l];
					}
				}
				else {
					if (bic_value > pre_bic_value) {
						pre_bic_value = bic_value;
						pre_amount = itr_cl_am;
						for (int l = 0; l < itr_cl_am; ++l) {
							pre_new_weights[l] = new_weights_mean[l];
							pre_new_shifts[l] = new_shifts[0][l];
							pre_new_scales[l] = new_scales[0][l];
						}
					}
				}
				for (int l = 0; l < itr_cl_am; ++l) {

					new_weights_mean[l] = 0;
					for (int t = 0; t < est_amount; ++t) {
						new_shifts[t][l] = 0;
						new_scales[t][l] = 0;
					}
				}

			}
			//cout << r << " " << j << " " << pre_amount<<endl;
			for (int l = 0; l < pre_amount; ++l) {
				//if(pre_amount==2)
				//cout <<"pre_new_scales[l] "<< pre_new_scales[l] << endl;
				if ((pre_new_weights[l] / d_u_new_n > min_p)
					&& (pre_new_scales[l] > 0)) {

					re_comp_shifts[r *  amount_window + j][l] = pre_new_scales[l];
					re_comp_scales[r * amount_window + j][l] = pre_new_scales[l];
				}
				pre_new_weights[l] = 0;
				pre_new_shifts[l] = 0;
				pre_new_scales[l] = 0;
			}
		}
	}

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n";
	for (int t = 0; t < (window_size + i_n_add)*(window_size + i_n_add); ++t) {
		delete[] new_g_ij[t];
		delete[] y_i_j[t];
		delete[] new_g_ij_0[t];
	}

	delete[] new_g_ij;
	delete[] y_i_j;
	delete[] new_g_ij_0;
	delete[] med_flag;
	delete[] new_weights_mean;
	delete[] pre_new_weights;
	delete[] pre_new_shifts;
	delete[] pre_new_scales;
	delete[] max_L_mass;

	for (j = 0; j < est_amount; ++j) {
		delete[] new_shifts[j];
		delete[] new_scales[j];
	}
	delete[] new_shifts;
	delete[] new_scales;

	for (j = 0; j < hyp_cl_amount; ++j) {
		delete[] new_med_sample[j];

	}
	delete[] new_med_sample;

}

//сетка к openmp

void  mixture_handler::create_mesh_opMP() {
	
	unsigned h_w = (window_size / 2);
	int amount_window = img_l_x / h_w - 1;
	//amount_trg = vert * horiz;
	unsigned k = hyp_cl_amount;
	long double summ = 0;
	long double summ1 = 0;
	long double new_n = window_size * window_size;
	const unsigned u_new_n = window_size * window_size;

	double   max_buf = 0;
	unsigned v_size = 0;
	unsigned counter = 0;
	bool     stop_flag = true;

	unsigned i, t, r;
	double l_b_summ, r_b_summ, left_bound, right_bound, remainder;;

	auto begin2 = std::chrono::steady_clock::now();
	while (stop_flag) {
		v_size = 260 / mesh_step;
		dist_mesh = new  mesh_elem[v_size];
		for (i = 0; i < v_size; i++)
			dist_mesh[i].knot_elem = i * mesh_step;

		for (i = 0; i < amount_window*amount_window; ++i) {
			for (t = 0; t < k; ++t) {
				if (re_comp_shifts[i][t] != 0) {
					left_bound = trunc(re_comp_shifts[i][t] / double(mesh_step));
					right_bound = ceil(re_comp_shifts[i][t] / double(mesh_step));
					if (left_bound >= v_size) {
						left_bound = v_size - 1;
						right_bound = v_size;
					}
					remainder = re_comp_shifts[i][t] - left_bound * double(mesh_step);
					if (remainder < double(mesh_step) / 2.0 + 0.055) {
						dist_mesh[unsigned(left_bound)].rights.push_back(re_comp_shifts[i][t]);
						dist_mesh[unsigned(left_bound)].rights_SKO.push_back(re_comp_scales[i][t]);
						++dist_mesh[unsigned(left_bound)].amount;
					}
					else {
						if (unsigned(right_bound) != v_size) {
							dist_mesh[unsigned(right_bound)].lefts.push_back(re_comp_shifts[i][t]);
							dist_mesh[unsigned(right_bound)].lefts_SKO.push_back(re_comp_scales[i][t]);
							++dist_mesh[unsigned(right_bound)].amount;
						}
						else {
							dist_mesh[unsigned(left_bound)].rights.push_back(re_comp_shifts[i][t]);
							dist_mesh[unsigned(left_bound)].rights_SKO.push_back(re_comp_scales[i][t]);
							++dist_mesh[unsigned(left_bound)].amount;
						}
					}
				}
			}
		}

		counter = 0;
		for (i = 0; i < v_size; i++)
			(dist_mesh[i].amount != 0) ? ++counter : counter += 0;

		if (counter == k)
			stop_flag = false;
		else {
			if (counter > k) {
				int middle;
				for (i = 0; i < v_size - 1; ++i) {
					if (dist_mesh[i].amount != 0 &&
						dist_mesh[i + 1].amount != 0) {
						l_b_summ = 0;
						r_b_summ = 0;
						for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++)
							l_b_summ += dist_mesh[i + 1].lefts[t];
						for (t = 0; t < dist_mesh[i + 1].rights.size(); t++)
							l_b_summ += dist_mesh[i + 1].rights[t];
						l_b_summ = l_b_summ / dist_mesh[i + 1].amount;

						for (t = 0; t < dist_mesh[i].rights.size(); t++)
							r_b_summ += dist_mesh[i].rights[t];
						for (t = 0; t < dist_mesh[i].lefts.size(); t++)
							r_b_summ += dist_mesh[i].lefts[t];
						r_b_summ = r_b_summ / dist_mesh[i].amount;

						if (l_b_summ - r_b_summ <= mesh_step / 2 + 0.555) {
							if (dist_mesh[i + 1].amount > dist_mesh[i].amount) {

								for (t = 0; t < dist_mesh[i].rights.size(); t++) {
									dist_mesh[i + 1].lefts.push_back(dist_mesh[i].rights[t]);
									dist_mesh[i + 1].lefts_SKO.push_back(dist_mesh[i].rights_SKO[t]);
								}
								for (t = 0; t < dist_mesh[i].lefts.size(); t++) {
									dist_mesh[i + 1].lefts.push_back(dist_mesh[i].lefts[t]);
									dist_mesh[i + 1].lefts_SKO.push_back(dist_mesh[i].lefts_SKO[t]);
								}
								dist_mesh[i + 1].amount += dist_mesh[i].amount;
								dist_mesh[i].rights.clear();
								dist_mesh[i].rights_SKO.clear();
								dist_mesh[i].lefts.clear();
								dist_mesh[i].lefts_SKO.clear();
								dist_mesh[i].amount = 0;
								--counter;

							}
							else {
								for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++) {
									dist_mesh[i].rights.push_back(dist_mesh[i + 1].lefts[t]);
									dist_mesh[i].rights_SKO.push_back(dist_mesh[i + 1].lefts_SKO[t]);
								}
								for (t = 0; t < dist_mesh[i + 1].rights.size(); t++) {
									dist_mesh[i].rights.push_back(dist_mesh[i + 1].rights[t]);
									dist_mesh[i].rights_SKO.push_back(dist_mesh[i + 1].rights_SKO[t]);
								}
								dist_mesh[i].amount += dist_mesh[i + 1].amount;
								dist_mesh[i + 1].lefts.clear();
								dist_mesh[i + 1].lefts_SKO.clear();
								dist_mesh[i + 1].rights.clear();
								dist_mesh[i + 1].rights_SKO.clear();
								dist_mesh[i + 1].amount = 0;
								--counter;
							}
						}
					}
					if (counter == k) {
						stop_flag = false;
						break;
					}
				}
				if (counter > k) {
					cout << "Houston, we have a problem! This is too much!" << endl;
					stop_flag = false;
				}
			}
			else {
				if (mesh_step > 1) {
					for (i = 0; i < v_size; i++) {
						dist_mesh[i].lefts.clear();
						dist_mesh[i].lefts_SKO.clear();
						dist_mesh[i].rights.clear();
						dist_mesh[i].rights_SKO.clear();
					}
					delete[] dist_mesh;
					mesh_step = mesh_step / 2;
				}
				else {
					cout << "Houston, we have a problem! This is not enough!" << endl;
					stop_flag = false;
				}
			}
		}
	}
	/*cout << " initial values - shifts " << "\n";
	for (i = 0; i < hyp_cl_amount; ++i)
		cout << mix_shift[i] << "  ";
	cout << "initial values - scales " << "\n";
	for (i = 0; i < hyp_cl_amount; ++i)
		cout << mix_scale[i] << "  ";
	cout << " " << "\n";*/
	hyp_cl_amount_mod = hyp_cl_amount;
	if (counter != k) {
		cout << " thought Mix components amount in the working window is less then in all picture. " << endl;
		cout << " thought Mix components amount = " << hyp_cl_amount << ", finded  Mix components amount " << counter << endl;
		equal_hyp_flag = false;
		hyp_cl_amount = counter;
		delete[] mix_shift;
		delete[] mix_scale;
		delete[] mix_weight;
		mix_shift = new double[hyp_cl_amount];
		mix_scale = new double[hyp_cl_amount];
		mix_weight = new double[hyp_cl_amount];
		for (i = 0; i < hyp_cl_amount; ++i)
			mix_weight[i] = 0.0;
	}

	r = 0;
	unsigned amount = 0;
	mix_prob = new double[hyp_cl_amount];

	for (i = 0; i < v_size; i++) {
		if (dist_mesh[i].amount != 0) {
			summ = 0;
			summ1 = 0;
			for (t = 0; t < dist_mesh[i].lefts.size(); ++t) {
				summ += dist_mesh[i].lefts[t];
				summ1 += dist_mesh[i].lefts_SKO[t];
			}
			for (t = 0; t < dist_mesh[i].rights.size(); ++t) {
				summ += dist_mesh[i].rights[t];
				summ1 += dist_mesh[i].rights_SKO[t];
			}
			amount += dist_mesh[i].amount;
			mix_prob[r] = dist_mesh[i].amount;
			mix_shift[r] = summ / double(dist_mesh[i].amount);
			mix_scale[r] = summ1 / double(dist_mesh[i].amount);
			++r;
		}
	}

	/*cout << "EM mix_shift values  1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
	{
		mix_prob[i] = mix_prob[i] / amount;
		cout << mix_prob[i] << "  ";
	}
	cout << endl;*/

	cout << "EM mix_scale values 1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	cout << "EM mix_shift values  1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	for (i = 0; i < amount_window*amount_window; ++i) {
		delete[] re_comp_shifts[i];
		delete[] re_comp_scales[i];
	}
	delete[] re_comp_shifts;
	delete[] re_comp_scales;


	for (i = 0; i < v_size; i++) {
		dist_mesh[i].lefts.clear();
		dist_mesh[i].lefts_SKO.clear();
		dist_mesh[i].rights.clear();
		dist_mesh[i].rights_SKO.clear();
	}
	delete[] dist_mesh;

}

//создание сетки получившихся компонент смеси - вынос в отдельную функцию

void  mixture_handler::create_mesh() {
	unsigned block_size = img_l_x / thr_nmb;
	unsigned h_w = (window_size / 2);
	unsigned vert = block_size / h_w;
	unsigned horiz = img_l_x / h_w - 1;
	amount_trg = vert * horiz;
	unsigned k = hyp_cl_amount;
	long double summ = 0;
	long double summ1 = 0;
	long double new_n = window_size * window_size;
	const unsigned u_new_n = window_size * window_size;
	
	double   max_buf = 0;
	unsigned v_size = 0;
	unsigned counter = 0;
	bool     stop_flag = true;
	
	unsigned i, t, r;
	double l_b_summ, r_b_summ, left_bound, right_bound, remainder;;

	auto begin2 = std::chrono::steady_clock::now();
	while (stop_flag) {
		v_size = 260 / mesh_step;
		dist_mesh = new  mesh_elem[v_size];
		for (i = 0; i < v_size; i++)
			dist_mesh[i].knot_elem = i * mesh_step;

		for (i = 0; i < thr_nmb*amount_trg - horiz; ++i) {
			for (t = 0; t < k; ++t) {
				if (re_comp_shifts[i][t] != 0) {
					left_bound = trunc(re_comp_shifts[i][t] / double(mesh_step));
					right_bound = ceil(re_comp_shifts[i][t] / double(mesh_step));
					if (left_bound >= v_size) {
						left_bound = v_size - 1;
						right_bound = v_size;
					}
					remainder = re_comp_shifts[i][t] - left_bound * double(mesh_step);
					if (remainder < double(mesh_step) / 2.0 + 0.055) {
						dist_mesh[unsigned(left_bound)].rights.push_back(re_comp_shifts[i][t]);
						dist_mesh[unsigned(left_bound)].rights_SKO.push_back(re_comp_scales[i][t]);
						++dist_mesh[unsigned(left_bound)].amount;
					}
					else {
						if (unsigned(right_bound) != v_size) {
							dist_mesh[unsigned(right_bound)].lefts.push_back(re_comp_shifts[i][t]);
							dist_mesh[unsigned(right_bound)].lefts_SKO.push_back(re_comp_scales[i][t]);
							++dist_mesh[unsigned(right_bound)].amount;
						}
						else {
							dist_mesh[unsigned(left_bound)].rights.push_back(re_comp_shifts[i][t]);
							dist_mesh[unsigned(left_bound)].rights_SKO.push_back(re_comp_scales[i][t]);
							++dist_mesh[unsigned(left_bound)].amount;
						}
					}
				}
			}
		}

		counter = 0;
		for (i = 0; i < v_size; i++)
			(dist_mesh[i].amount != 0) ? ++counter : counter += 0;

		if (counter == k)
			stop_flag = false;
		else {
			if (counter > k) {
				int middle;
				for (i = 0; i < v_size - 1; ++i) {
					if (dist_mesh[i].amount != 0 &&
						dist_mesh[i + 1].amount != 0) {
						l_b_summ = 0;
						r_b_summ = 0;
						for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++)
							l_b_summ += dist_mesh[i + 1].lefts[t];
						for (t = 0; t < dist_mesh[i + 1].rights.size(); t++)
							l_b_summ += dist_mesh[i + 1].rights[t];
						l_b_summ = l_b_summ / dist_mesh[i + 1].amount;

						for (t = 0; t < dist_mesh[i].rights.size(); t++)
							r_b_summ += dist_mesh[i].rights[t];
						for (t = 0; t < dist_mesh[i].lefts.size(); t++)
							r_b_summ += dist_mesh[i].lefts[t];
						r_b_summ = r_b_summ / dist_mesh[i].amount;

						if (l_b_summ - r_b_summ <= mesh_step / 2 + 0.555) {
							if (dist_mesh[i + 1].amount > dist_mesh[i].amount) {

								for (t = 0; t < dist_mesh[i].rights.size(); t++) {
									dist_mesh[i + 1].lefts.push_back(dist_mesh[i].rights[t]);
									dist_mesh[i + 1].lefts_SKO.push_back(dist_mesh[i].rights_SKO[t]);
								}
								for (t = 0; t < dist_mesh[i].lefts.size(); t++) {
									dist_mesh[i + 1].lefts.push_back(dist_mesh[i].lefts[t]);
									dist_mesh[i + 1].lefts_SKO.push_back(dist_mesh[i].lefts_SKO[t]);
								}
								dist_mesh[i + 1].amount += dist_mesh[i].amount;
								dist_mesh[i].rights.clear();
								dist_mesh[i].rights_SKO.clear();
								dist_mesh[i].lefts.clear();
								dist_mesh[i].lefts_SKO.clear();
								dist_mesh[i].amount = 0;
								--counter;

							}
							else {
								for (t = 0; t < dist_mesh[i + 1].lefts.size(); t++) {
									dist_mesh[i].rights.push_back(dist_mesh[i + 1].lefts[t]);
									dist_mesh[i].rights_SKO.push_back(dist_mesh[i + 1].lefts_SKO[t]);
								}
								for (t = 0; t < dist_mesh[i + 1].rights.size(); t++) {
									dist_mesh[i].rights.push_back(dist_mesh[i + 1].rights[t]);
									dist_mesh[i].rights_SKO.push_back(dist_mesh[i + 1].rights_SKO[t]);
								}
								dist_mesh[i].amount += dist_mesh[i + 1].amount;
								dist_mesh[i + 1].lefts.clear();
								dist_mesh[i + 1].lefts_SKO.clear();
								dist_mesh[i + 1].rights.clear();
								dist_mesh[i + 1].rights_SKO.clear();
								dist_mesh[i + 1].amount = 0;
								--counter;
							}
						}
					}
					if (counter == k) {
						stop_flag = false;
						break;
					}
				}
				if (counter > k) {
					cout << "Houston, we have a problem! This is too much!" << endl;
					stop_flag = false;
				}
			}
			else {
				if (mesh_step > 1) {
					for (i = 0; i < v_size; i++) {
						dist_mesh[i].lefts.clear();
						dist_mesh[i].lefts_SKO.clear();
						dist_mesh[i].rights.clear();
						dist_mesh[i].rights_SKO.clear();
					}
					delete[] dist_mesh;
					mesh_step = mesh_step / 2;
				}
				else {
					cout << "Houston, we have a problem! This is not enough!" << endl;
					stop_flag = false;
				}
			}
		}
	}
	cout << " initial values - shifts " << "\n";
	for (i = 0; i < hyp_cl_amount; ++i)
		cout << mix_shift[i] << "  ";
	cout << "initial values - scales " << "\n";
	for (i = 0; i < hyp_cl_amount; ++i)
		cout << mix_scale[i] << "  ";
	cout << " " << "\n";
	hyp_cl_amount_mod = hyp_cl_amount;
	if (counter != k) {
		cout << " thought Mix components amount in the working window is less then in all picture. " << endl;
		cout << " thought Mix components amount = " << hyp_cl_amount << ", finded  Mix components amount " << counter << endl;
		equal_hyp_flag = false;
		hyp_cl_amount = counter;
		delete[] mix_shift;
		delete[] mix_scale;
		delete[] mix_weight;
		mix_shift = new double[hyp_cl_amount];
		mix_scale = new double[hyp_cl_amount];
		mix_weight = new double[hyp_cl_amount];
		for (i = 0; i < hyp_cl_amount; ++i)
			mix_weight[i] = 0.0;
	}

	r = 0;
	unsigned amount = 0;
	mix_prob = new double[hyp_cl_amount];

	for (i = 0; i < v_size; i++) {
		if (dist_mesh[i].amount != 0) {
			summ = 0;
			summ1 = 0;
			for (t = 0; t < dist_mesh[i].lefts.size(); ++t) {
				summ += dist_mesh[i].lefts[t];
				summ1 += dist_mesh[i].lefts_SKO[t];
			}
			for (t = 0; t < dist_mesh[i].rights.size(); ++t) {
				summ += dist_mesh[i].rights[t];
				summ1 += dist_mesh[i].rights_SKO[t];
			}
			amount += dist_mesh[i].amount;
			mix_prob[r] = dist_mesh[i].amount;
			mix_shift[r] = summ / double(dist_mesh[i].amount);
			mix_scale[r] = summ1 / double(dist_mesh[i].amount);
			++r;
		}
	}

	cout << "EM mix_shift values  1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
	{
		mix_prob[i] = mix_prob[i] / amount;
		cout << mix_prob[i] << "  ";
	}
	cout << endl;

	cout << "EM mix_scale values 1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	cout << "EM mix_frequency values  1 step:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	for (i = 0; i < thr_nmb*amount_trg - horiz; ++i) {
		delete[] re_comp_shifts[i];
		delete[] re_comp_scales[i];
	}
	delete[] re_comp_shifts;
	delete[] re_comp_scales;
	

	for (i = 0; i < v_size; i++) {
		dist_mesh[i].lefts.clear();
		dist_mesh[i].lefts_SKO.clear();
		dist_mesh[i].rights.clear();
		dist_mesh[i].rights_SKO.clear();
	}
	delete[] dist_mesh;

}

//раскраска картики - em алгоритм , std:: thread version

void mixture_handler::optimal_redraw() {
	window_size = 86;
	//window_size = window_size/ 2;
	/////////////////////////////////
	delete[] mix_scale;
	delete[] mix_shift;
	hyp_cl_amount = 6;
	mix_scale = new double[hyp_cl_amount];
	mix_shift = new double[hyp_cl_amount];

	for (int i = 0; i < hyp_cl_amount; ++i) {
		mix_scale[i] = gen_image->get_scale()[i];
		mix_shift[i] = gen_image->get_shift()[i];
	}
	/////////////////////////////////
	//window_size = window_size * 2;
	unsigned k = hyp_cl_amount;
	thr_nmb = 4;
	unsigned block_size = img_l_x / thr_nmb;
	cout << "block_size " << block_size << endl;
	if (block_size >= 2 * window_size)
		window_size = 1 * window_size;
	else {
		window_size = block_size;
	}
	long double summ = 0;
	long double summ1 = 0;


	unsigned u_new_n = window_size * window_size;
	unsigned   **buf_comp_weights = new unsigned*[thr_nmb];
	unsigned i, t;
	auto mix_part_redraw = [&](unsigned i) {
		long double summ = 0;

		double last_cur_max = 0;
		long double pix_buf = 0;
		double cur_max = 0;
		long double** new_g_ij = new long double*[u_new_n];
		long double** new_g_ij_0 = new long double*[u_new_n];
		long double * new_weights = new long double[k];
		long double * buf_new_weights = new long double[k];
		bool stop_flag = true;
		double buf_max = 0;
		unsigned idx_i = 0;
		unsigned idx_j = 0;
		unsigned idx_max = 0;
		unsigned bl_w = block_size / window_size;
		unsigned im_w = img_l_x / window_size;
		unsigned x_min, y_min, l, j, t, r;
		x_min = i * block_size - window_size;
		const double sq_pi = sqrt(2 * pi);
		if (i == thr_nmb - 1 && block_size*thr_nmb < img_l_x)
			block_size += img_l_x - block_size * thr_nmb;
		for (l = 0; l < u_new_n; ++l) {
			new_g_ij[l] = new long double[k];
			new_g_ij_0[l] = new long double[k];
			for (t = 0; t < k; t++) {
				new_g_ij[l][t] = 0;
				new_g_ij_0[l][t] = 0;
			}
		}
		for (l = 0; l < k; ++l)
			//new_weights[l] = mix_prob[l];
			new_weights[l] = 1.0 / double(k);

		unsigned hor_rem = img_l_x - window_size * im_w;
		unsigned vert_rem = block_size - window_size * bl_w;
		if (vert_rem > window_size) {
			bl_w += 1;
			vert_rem -= window_size;
		}
		unsigned r_bound, y_l, x_l;

		x_l = window_size;
		for (r = 0; r < bl_w + 1; ++r) {
			x_min += window_size;
			/*if(i == thr_nmb - 1 && r == bl_w)*/
			if (r == bl_w)
				x_l = vert_rem;
			y_l = window_size;
			for (j = 0; j < im_w + 1; ++j) {
				y_min = j * window_size;
				stop_flag = true;
				cur_max = 0;
				if (j == im_w)
					y_l = hor_rem;

				r_bound = y_l * x_l;
				while (stop_flag) {
					for (l = 0; l < r_bound; ++l) {
						summ = 0;
						idx_i = x_min + l / y_l;
						idx_j = y_min + l % y_l;
						pix_buf = my_picture[idx_i][idx_j];
						for (t = 0; t < k; ++t)
							summ += new_weights[t] * (1 / (mix_scale[t] * sq_pi))*exp(-((pix_buf
								- mix_shift[t])*(pix_buf - mix_shift[t])) / (2.0 * mix_scale[t] * mix_scale[t]));

						for (t = 0; t < k; ++t) {
							if (l == 0)
								buf_new_weights[t] = 0;
							new_g_ij[l][t] = new_weights[t] * (1 / (mix_scale[t] * sq_pi*summ))*exp(-((pix_buf
								- mix_shift[t])*(pix_buf - mix_shift[t])) / (2.0 * mix_scale[t] * mix_scale[t]));
							buf_new_weights[t] += new_g_ij[l][t];
							if (l == r_bound - 1)
								new_weights[t] = buf_new_weights[t] / double(r_bound);
							if (cur_max < abs(new_g_ij[l][t] - new_g_ij_0[l][t]))
								cur_max = abs(new_g_ij[l][t] - new_g_ij_0[l][t]);
							new_g_ij_0[l][t] = new_g_ij[l][t];
						}
					}

					if (stop_flag != false) {
						if (cur_max != 0)
							last_cur_max = cur_max;
						(cur_max < accuracy) ? stop_flag = false : cur_max = 0;
					}
				}

				for (t = 0; t < r_bound; ++t) {
					idx_i = x_min + t / y_l;
					idx_j = y_min + t % y_l;
					buf_max = new_g_ij[t][0];
					idx_max = 0;
					for (l = 0; l < k; ++l) {
						if (t == 0)
							//new_weights[l] = mix_prob[l];
							new_weights[l] = 1.0 / double(k);
						if (buf_max < new_g_ij[t][l]) {
							buf_max = new_g_ij[t][l];
							idx_max = l;
						}
						new_g_ij_0[t][l] = 0;
					}
					class_flag[idx_i][idx_j] = idx_max + 1;
					buf_comp_weights[i][idx_max] += 1;
				}
			}

		}
		for (t = 0; t < u_new_n; ++t) {
			delete[] new_g_ij[t];
			delete[] new_g_ij_0[t];
		}
		delete[] new_g_ij;
		delete[] new_g_ij_0;
		delete[] new_weights;
		delete[] buf_new_weights;
	};

	for (i = 0; i < thr_nmb; ++i) {
		buf_comp_weights[i] = new unsigned[hyp_cl_amount];
		for (t = 0; t < hyp_cl_amount; ++t)
			buf_comp_weights[i][t] = 0;
		mix_weight[t] = 0;
	}

	auto begin1 = std::chrono::steady_clock::now();
	std::thread threadObj1(mix_part_redraw, 0);
	std::thread threadObj2(mix_part_redraw, 1);
	std::thread threadObj3(mix_part_redraw, 2);
	std::thread threadObj4(mix_part_redraw, 3);

	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();

	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << "\n" << "\n" << "\n";
	double b = 0;
	for (i = 0; i < thr_nmb; ++i) {
		for (t = 0; t < hyp_cl_amount; ++t)
			mix_weight[t] += buf_comp_weights[i][t] / double(img_l_x * img_l_x);
	}
	cout << endl;
	for (t = 0; t < hyp_cl_amount; ++t) {
		cout << mix_weight[t] << " ";
		b += mix_weight[t];
	}
	cout << "\n" << b << endl;
	for (i = 0; i < thr_nmb; ++i)
		delete[] buf_comp_weights[i];
}

//раскраска картики - em алгоритм , openMP version

void mixture_handler::optimal_redraw_opMP() {
	window_size = 10;
   // window_size = window_size * 2;
	int add_amount_x = img_l_x% window_size;
	int add_amount_y = img_l_y % window_size;
	int amount_window_x = img_l_x / window_size ;
	int amount_window_y = img_l_y / window_size;
	unsigned u_new_n = (window_size + add_amount_x) *(window_size + add_amount_y);
	/////////////////////////////////
	delete[] mix_scale;
	delete[] mix_shift;
	hyp_cl_amount = gen_image->get_class_amount()+1;
	mix_scale = new double[hyp_cl_amount];
	mix_shift = new double[hyp_cl_amount];

	for (int i = 0; i < hyp_cl_amount; ++i) {
		mix_scale[i] = gen_image->get_scale()[i];
		mix_shift[i] = gen_image->get_shift()[i];
	}
	/////////////////////////////////
	
	auto begin1 = std::chrono::steady_clock::now();
	#pragma omp parallel
	{
		int loc_window_size = window_size;
		int loc_hyp_cl_amount = hyp_cl_amount;
		int x_l = loc_window_size;
		int y_l = loc_window_size;

		int itr, x_min, y_min, j, loc_u_new_n, t, l;
		double** new_g_ij = new double*[u_new_n];
		double** new_g_ij_0 = new double*[u_new_n];
		double * new_weights = new  double[loc_hyp_cl_amount];
		double * new_shifts = new  double[loc_hyp_cl_amount];
		double * new_scales = new  double[loc_hyp_cl_amount];
		double * buf_new_weights = new  double[loc_hyp_cl_amount];
		
		for (l = 0; l < u_new_n; ++l) {
			new_g_ij[l] = new double[loc_hyp_cl_amount];
			new_g_ij_0[l] = new double[loc_hyp_cl_amount];
			for (t = 0; t < loc_hyp_cl_amount; t++) {
				new_g_ij[l][t] = 0;
				new_g_ij_0[l][t] = 0;
			}
		}
		for (l = 0; l < loc_hyp_cl_amount; ++l) {
			new_weights[l] = 1.0 / double(loc_hyp_cl_amount);
			new_shifts[l] = mix_shift[l];
			new_scales[l] = mix_scale[l];
		}
		//std::function<double(double , double , double , double )> sum_func = [&](double shift, double scale, double weight, double pix_buf) { return 0; };
		
		double summ = 0;
		const double sq_pi = sqrt(2 * pi);
		double pix_buf,cur_max;
		/*if (mixture_type == "normal")
			 sum_func = [&](double shift, double scale, double weight, double pix_buf) {
			
				return weight * (1 / (scale * sqrt(2 * pi)))*exp(-((pix_buf
					- shift)*(pix_buf - shift)) / (2.0 * scale * scale));
			};
		else {
			if (mixture_type == "lognormal")
				 sum_func = [&](double shift, double scale, double weight, double pix_buf) {

				return weight * (1 / (scale * sqrt(2 * pi)*pix_buf))*exp(-((log(pix_buf)
					- shift)*(log(pix_buf) - shift)) / (2.0 * scale * scale));
				};
			else {
				if (mixture_type == "rayleigh")
					sum_func = [&](double shift, double scale, double weight, double pix_buf) {

					return weight * (pix_buf / (scale * scale))*exp(-(pix_buf*pix_buf) / (2.0 * scale * scale));
				};
			}
		}*/
		double last_cur_max = 0;

		double buf_max = 0;
		bool stop_flag = true;
		unsigned idx_max = 0;
		#pragma omp for
		for (int r = 0; r < amount_window_x; ++r) {
			x_min = r * loc_window_size;
			if (r < amount_window_x - 1)
				x_l = loc_window_size;
			else
				x_l = loc_window_size + add_amount_x;
			for (j = 0; j < amount_window_y; ++j) {

				y_min = j * loc_window_size;
				if (j < amount_window_y - 1)
					y_l = loc_window_size;
				else
					y_l = loc_window_size + add_amount_y;
				

				itr = 0;
				stop_flag = true;
				cur_max = 0;
				loc_u_new_n = y_l * x_l;
				
				while (stop_flag && (itr < 500)) {
					++itr;
					
					for (l = 0; l < loc_u_new_n; ++l) {
						summ = 0;
						pix_buf = my_picture[x_min + l / y_l][y_min + l % y_l];
						for (t = 0; t < loc_hyp_cl_amount; ++t) {
						//if (mixture_type == "normal")
							//summ += sum_func(mix_shift[t], mix_scale[t], new_weights[t], pix_buf);
							//cout << sum_func(mix_shift[t], mix_scale[t], new_weights[t], pix_buf)<< endl;
								/*summ += new_weights[t] * (1 / (mix_scale[t] * sq_pi))*exp(-((pix_buf
									- mix_shift[t])*(pix_buf - mix_shift[t])) / (2.0 * mix_scale[t] * mix_scale[t]));*/
							/*else {
								if (mixture_type == "rayleigh")*/
									/*summ += new_weights[t] * (pix_buf / (mix_scale[t] * mix_scale[t]))*exp(-((pix_buf
										)*(pix_buf )) / (2.0 * mix_scale[t] * mix_scale[t]));*/
							/*}*/
							//else {
							//if (mixture_type == "lognormal")
								summ += new_weights[t] * (1 / (mix_scale[t] * sq_pi*pix_buf))*exp(-((log(pix_buf)
									- mix_shift[t])*(log(pix_buf) - mix_shift[t])) / (2.0 * mix_scale[t] * mix_scale[t]));
						//}

						}

						for (t = 0; t < loc_hyp_cl_amount; ++t) {
								if (l == 0)
									buf_new_weights[t] = 0;
								//if (mixture_type == "normal")
									/*new_g_ij[l][t] = new_weights[t] * (1 / (mix_scale[t] * sq_pi*summ))*exp(-((pix_buf
										- mix_shift[t])*(pix_buf - mix_shift[t])) / (2.0 * mix_scale[t] * mix_scale[t]));*/
								//new_g_ij[l][t] = sum_func(mix_shift[t], mix_scale[t], new_weights[t], pix_buf) / summ; 
								/*else {
									if (mixture_type == "rayleigh")*/
										/*new_g_ij[l][t] = new_weights[t] * (pix_buf / (mix_scale[t] * mix_scale[t]))*exp(-(pix_buf
											*pix_buf) / (2.0 * mix_scale[t] * mix_scale[t]));*/
								/*}*/
								///else {
								//if (mixture_type == "lognormal")
									new_g_ij[l][t] = new_weights[t] * (1 / (mix_scale[t] * sq_pi*summ*pix_buf))*exp(-((log(pix_buf)
										- mix_shift[t])*(log(pix_buf) - mix_shift[t])) / (2.0 * mix_scale[t] * mix_scale[t]));
								//}
								buf_new_weights[t] += new_g_ij[l][t];
								if (l == loc_u_new_n - 1)
									new_weights[t] = buf_new_weights[t] / double(loc_u_new_n);
								if (cur_max < abs(new_g_ij[l][t] - new_g_ij_0[l][t]))
									cur_max = abs(new_g_ij[l][t] - new_g_ij_0[l][t]);
								new_g_ij_0[l][t] = new_g_ij[l][t];
							}
						
					}
					
					if (stop_flag != false) {
						if (cur_max != 0)
							last_cur_max = cur_max;
						(cur_max < accuracy) ? stop_flag = false : cur_max = 0;
					}
				}
				#pragma omp critical
				{

					for (t = 0; t < loc_u_new_n; ++t) {
						buf_max = new_g_ij[t][0];
						idx_max = 0;
						for (l = 0; l < loc_hyp_cl_amount; ++l) {
							if (t == 0)
								//new_weights[l] = mix_prob[l];
								new_weights[l] = 1.0 / double(loc_hyp_cl_amount);
							if (buf_max < new_g_ij[t][l]) {
								buf_max = new_g_ij[t][l];
								idx_max = l;
							}
							new_g_ij_0[t][l] = 0;
						}
						class_flag[x_min + t / y_l][y_min + t % y_l] = idx_max + 1;
					}

				//cout << "x_min " << x_min << endl;
				}
			}

		}

		for (t = 0; t < u_new_n; ++t) {
			delete[] new_g_ij[t];
			delete[] new_g_ij_0[t];
		}
		delete[] new_g_ij;
		delete[] new_g_ij_0;
		delete[] new_weights;
		delete[] buf_new_weights;
		delete[] new_shifts;
		delete[] new_scales;
	}
	
	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() <<endl;
}

//раскраска с использованием критерия колмогорова

void mixture_handler::kolmogorov_optimal_redraw() {
	int length_ = 5;
	int half_step = length_ / 2;
	int iters_i_x = img_l_x / half_step;
	int iters_i_y = img_l_y / half_step;
	while ((iters_i_x*half_step + length_ )> img_l_x)
		iters_i_x--;
	while ((iters_i_y*half_step + length_) > img_l_y)
		iters_i_y--;
	cout << "iters_i*half_step + length_ " << iters_i_x * half_step + length_ << endl;
	int x_l, y_l, idx_class;
	double* buf_img = new double[4 * length_*length_];
	auto begin1 = std::chrono::steady_clock::now();
	auto L_max_calculation = [&](double* data, int data_size, int iter, double mix_shift, double mix_scale, double* max_L_mass) {
		double buf_max_l = 0;
		bool flag = false;
		double B;
		for (int m = 0; m < data_size; m++) {
			B = 0;
			if (mix_scale != 0) {
				if (mixture_type == "normal")
					B = (1.0 / (mix_scale))*
					exp(-(pow(data[m] - mix_shift, 2)) /
					(2.0 * mix_scale * mix_scale));
				if (mixture_type == "lognormal")
					B = (1.0 / (mix_scale*data[m]))*
					exp(-(pow(log(data[m]) - mix_shift, 2)) /
					(2.0 * mix_scale * mix_scale));
				if (mixture_type == "rayleigh")
					B = (data[m] / pow(mix_scale, 2))*
					exp(-(pow(data[m], 2)) /
					(2.0 * mix_scale * mix_scale));
			}
			else {
				flag = true;
				break;
			}

			if (B > 0)
				buf_max_l = buf_max_l + log(B / (data_size));
		}

		max_L_mass[iter] = buf_max_l;
	};

	auto kolmogorov_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
		unsigned i, j;
		int intervals_amount = 60;
		int k, buf_intervals_amount;
		bool flag = true;
		double max_d_n, buf_d_n, F_n_curr;
		int* flag_mass = new int[hyp_cl_amount];
		double* stats_mass = new double[hyp_cl_amount];
		int flag_summ = 0;
		int flag_idx = 0;
		double* nu_i = new double[intervals_amount];
		double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
		double min_value = find_k_stat(data, data_size, 0);
		double len_interval = (max_value - min_value) / intervals_amount;
		double *max_L_mass = new double[hyp_cl_amount];
		for (i = 0; i < intervals_amount; ++i)
			nu_i[i] = 0;
		for (i = 0; i < data_size; ++i) {
			k = 0;
			flag = true;
			while (flag) {
				if (((len_interval * k + min_value) <= data[i]) && ((len_interval * (k + 1) + min_value) > data[i]))
					flag = false;
				else
					k++;
			}
			nu_i[k] = nu_i[k] + 1;
		}

		double dn_bound;
		if(length_ == 5)
			dn_bound = 0.264;
		else
			dn_bound = 0.134;
		//double dn_bound = 0.238;
		for (i = 0; i < hyp_cl_amount; ++i) {
			F_n_curr = 0;
			max_d_n = 0;
			if (mixture_type == "normal")
				max_d_n = cdf(normal(mix_shift[i], mix_scale[i]), min_value);
			if (mixture_type == "rayleigh")
				max_d_n = cdf(rayleigh(mix_scale[i]), min_value);
			for (j = 0; j < intervals_amount; ++j) {
				F_n_curr += nu_i[j] / data_size;
				if (j != intervals_amount - 1) {
					for (k = 0; k < 2; ++k) {
						if (mixture_type == "normal")
							buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), (len_interval *(j + k) + min_value)) - F_n_curr);
						if (mixture_type == "rayleigh")
							buf_d_n = abs(cdf(rayleigh(mix_scale[i]), (len_interval *(j + k) + min_value)) - F_n_curr);
						if (buf_d_n > max_d_n)
							max_d_n = buf_d_n;
					}
				}
				else {
					if (mixture_type == "normal")
						buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), (len_interval *(j + 1) + min_value)) - 1);
					if (mixture_type == "rayleigh")
						buf_d_n = abs(cdf(rayleigh(mix_scale[i]), (len_interval *(j + 1) + min_value)) - 1);
					if (buf_d_n > max_d_n)
						max_d_n = buf_d_n;
				}
			}

			stats_mass[i] = max_d_n;
			if (max_d_n < dn_bound)
				flag_mass[i] = 1;
			else
				flag_mass[i] = 0;
		}
		for (i = 0; i < hyp_cl_amount; ++i) {
			flag_summ += flag_mass[i];
			if (flag_mass[i] == 1)
				flag_idx = i;
		}
		if (flag_summ > 1) {
			for (int i = 0; i < hyp_cl_amount; ++i) 
				L_max_calculation( data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

			int beg_idx = 0;
			int max_idx = 0;

			for (int m = beg_idx; m < hyp_cl_amount; ++m) {
				if (max_L_mass[max_idx] < max_L_mass[m])
					max_idx = m;
			}
			flag_summ = 1;
			flag_idx = max_idx;
		}
		delete[] flag_mass;
		delete[] stats_mass;
		delete[] max_L_mass;
		delete[] nu_i;
		if (flag_summ == 1)
			return flag_idx;
		else return -1;
	};
	auto max_likehood_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
		unsigned i, j;
		int intervals_amount = 60;
		int k, buf_intervals_amount;
		bool flag = true;
		double max_d_n, buf_d_n, F_n_curr;
		//int* flag_mass = new int[hyp_cl_amount];
		//double* stats_mass = new double[hyp_cl_amount];
		int flag_summ = 0;
		int flag_idx = 0;
		/*double* nu_i = new double[intervals_amount];
		double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
		double min_value = find_k_stat(data, data_size, 0);
		double len_interval = (max_value - min_value) / intervals_amount;*/
		double *max_L_mass = new double[hyp_cl_amount];
		
		
			for (int i = 0; i < hyp_cl_amount; ++i)
				L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

			int beg_idx = 0;
			int max_idx = 0;

			for (int m = beg_idx; m < hyp_cl_amount; ++m) {
				if (max_L_mass[max_idx] < max_L_mass[m])
					max_idx = m;
			}
			flag_summ = 1;
			flag_idx = max_idx;
	
		
		delete[] max_L_mass;
		
		if (flag_summ == 1)
			return flag_idx;
		else return -1;
	};
	auto kolmogorov_stats2 = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
		unsigned i, j;
		int intervals_amount = 60;
		int k, buf_intervals_amount;
		bool flag = true;
		double max_d_n, buf_d_n, F_n_curr;
		int* flag_mass = new int[hyp_cl_amount];
		double* stats_mass = new double[hyp_cl_amount];
		int flag_summ = 0;
		int flag_idx = 0;
		
		double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
		double min_value = find_k_stat(data, data_size, 0);
		double len_interval = (max_value - min_value) / intervals_amount;
		double *max_L_mass = new double[hyp_cl_amount];


		
		double temp;
		for (int i = 0; i < data_size - 1; i++) {
			for (int j = 0; j < data_size - i - 1; j++) {
				if (data[j] > data[j + 1]) {
					// меняем элементы местами
					temp = data[j];
					data[j] = data[j + 1];
					data[j + 1] = temp;
				}
			}
		}
		double dn_bound;
		if (length_ == 5)
			dn_bound = 0.264;
		else
			dn_bound = 0.161;
		//double dn_bound = 0.238;
		for (i = 0; i < hyp_cl_amount; ++i) {
			F_n_curr = 0;
			max_d_n = 0;
			if (mixture_type == "normal")
				max_d_n = cdf(normal(mix_shift[i], mix_scale[i]), min_value);
			if (mixture_type == "rayleigh")
				max_d_n = cdf(rayleigh(mix_scale[i]), min_value);
			if (mixture_type == "lognormal")
				max_d_n = cdf(lognormal(mix_shift[i], mix_scale[i]), min_value);
			for (j = 1; j < data_size; ++j) {
				F_n_curr +=1.0 / data_size;
				if (j != data_size - 1) {
					for (k = 0; k < 2; ++k) {
						if (mixture_type == "normal")
							buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - (F_n_curr-k* 1.0 / data_size));
						if (mixture_type == "lognormal")
							buf_d_n = abs(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]) - (F_n_curr - k * 1.0 / data_size));
						if (mixture_type == "rayleigh")
							buf_d_n = abs(cdf(rayleigh(mix_scale[i]), data[j]) - (F_n_curr - k * 1.0 / data_size));
						if (buf_d_n > max_d_n)
							max_d_n = buf_d_n;
					}
				}
				else {
					if (mixture_type == "normal")
						buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - 1);
					if (mixture_type == "lognormal")
						buf_d_n = abs(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]) - 1);
					if (mixture_type == "rayleigh")
						buf_d_n = abs(cdf(rayleigh(mix_scale[i]), data[j]) - 1);
					if (buf_d_n > max_d_n)
						max_d_n = buf_d_n;
				}
			}

			stats_mass[i] = max_d_n;
			if (max_d_n < dn_bound)
				flag_mass[i] = 1;
			else
				flag_mass[i] = 0;
		}
		for (i = 0; i < hyp_cl_amount; ++i) {
			flag_summ += flag_mass[i];
			if (flag_mass[i] == 1)
				flag_idx = i;
		}
		if (flag_summ > 1) {
			for (int i = 0; i < hyp_cl_amount; ++i)
				L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

			int beg_idx = 0;
			int max_idx = 0;

			for (int m = beg_idx; m < hyp_cl_amount; ++m) {
				if (max_L_mass[max_idx] < max_L_mass[m])
					max_idx = m;
			}
			flag_summ = 1;
			flag_idx = max_idx;
		}
		delete[] flag_mass;
		delete[] stats_mass;
		delete[] max_L_mass;
		
		if (flag_summ == 1)
			return flag_idx;
		else return -1;
	};
	auto chi_square_stats = [&](double* data, int data_size) {
		int i, j, k, mix_params_amount, buf_intervals_amount;
		double chi_stat, teor_nu, quant_chi;
		bool flag = true;
		int intervals_amount =30;
		double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
		double min_value = find_k_stat(data, data_size, 0) - 1;
		double len_interval = (max_value - min_value) / intervals_amount;
		if (mixture_type == "normal")
			mix_params_amount = 2;
		if (mixture_type == "rayleigh")
			mix_params_amount = 1;
		if (mixture_type == "lognormal")
			mix_params_amount = 2;
		double* nu_i = new double[intervals_amount];
		double* interval_bounds = new double[intervals_amount];
		double* nu_i_bounds = new double[intervals_amount];
		double *max_L_mass = new double[hyp_cl_amount];
		int* flag_mass = new int[hyp_cl_amount];
		
		for (i = 0; i < intervals_amount; ++i)
			nu_i[i] = 0;
		for (i = 0; i < data_size; ++i) {
			k = 0;
			flag = true;
			while (flag) {
				if (((len_interval * k + min_value) <= data[i]) && ((len_interval * (k + 1) + min_value) > data[i]))
					flag = false;
				else
					k++;
			}
			nu_i[k] = nu_i[k] + 1;
		}
		flag = true;
		int r = data_size;
		int buf_nu = 0;
		buf_intervals_amount = 0;
		for (int j = 0; j < intervals_amount; ++j) {
			//cout << "nu_i[k] " << nu_i[j] << endl;
			if ((nu_i[j] > 5) && flag) {
				if (j < intervals_amount - 1) {
					if ((r - nu_i[j] > 5)) {
						interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
						nu_i_bounds[buf_intervals_amount] = nu_i[j];
						r -= nu_i[j];
						buf_intervals_amount++;
					}
					else {
						flag = false;
						buf_nu = nu_i[j];
						r -= nu_i[j];
					}
				}
				else {
					interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
					nu_i_bounds[buf_intervals_amount] = nu_i[j];
					r -= nu_i[j];
					buf_intervals_amount++;
				}
			}
			else {
				if (flag) {
					buf_nu = nu_i[j];
					r -= nu_i[j];
					flag = false;
				}
				else {
					buf_nu += nu_i[j];
					r -= nu_i[j];
				}
				if (buf_nu > 5) {
					if (j < intervals_amount - 1) {
						if (r > 5)
						{
							interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
							nu_i_bounds[buf_intervals_amount] = buf_nu;
							buf_intervals_amount++;
							buf_nu = 0;

							flag = true;
						}
					}
					else {
						//cout << "j " << j << " buf_nu " << buf_nu << endl;
						interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
						nu_i_bounds[buf_intervals_amount] = buf_nu;
						buf_intervals_amount++;
						buf_nu = 0;

						flag = true;
					}
				}
			}

		}

		int flag_summ = 0;
		int flag_idx = 0;

		if (buf_intervals_amount > mix_params_amount + 1) {
			//cout << "+" << endl;
			for (i = 0; i < hyp_cl_amount; ++i) {
				chi_stat = 0;

				for (j = 0; j < buf_intervals_amount; ++j) {
					if (j == 0) {
						if (mixture_type == "normal")
							teor_nu = cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j]);
						if (mixture_type == "lognormal")
							teor_nu = cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j]);
						if (mixture_type == "rayleigh")
							teor_nu = cdf(rayleigh(mix_scale[i]), interval_bounds[j]);
					}
					else {
						if (j != buf_intervals_amount - 1) {
							if (mixture_type == "normal")
								teor_nu = cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j])
								- cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
							if (mixture_type == "lognormal")
								teor_nu = cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j])
								- cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
							if (mixture_type == "rayleigh")
								teor_nu = cdf(rayleigh(mix_scale[i]), interval_bounds[j])
								- cdf(rayleigh(mix_scale[i]), interval_bounds[j - 1]);
						}
						else {
							if (mixture_type == "normal")
								teor_nu = 1 - cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
							if (mixture_type == "lognormal")
								teor_nu = 1 - cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
							if (mixture_type == "rayleigh")
								teor_nu = 1 - cdf(rayleigh(mix_scale[i]), interval_bounds[j - 1]);
						}
					}
					//cout << "teor_nu " << teor_nu << endl;
					teor_nu = teor_nu * (data_size);
					chi_stat += (nu_i_bounds[j] - teor_nu)* (nu_i_bounds[j] - teor_nu) / teor_nu;

				}
				quant_chi = quantile(chi_squared(buf_intervals_amount - 1 - mix_params_amount), 0.99);
				//cout << "classs " << i << ": chi_stat - " << chi_stat << " quant_chi = " << quant_chi << endl;
				if (chi_stat < quant_chi)
					flag_mass[i] = 1;
				else
					flag_mass[i] = 0;
			}

			for (i = 0; i < hyp_cl_amount; ++i) {
				flag_summ += flag_mass[i];
				if (flag_mass[i] == 1)
					flag_idx = i;
			}
		}
		else 
			flag_summ = 0;
		
		if (flag_summ > 1) {
			for (int i = 0; i < hyp_cl_amount; ++i) 
				L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);
				
			int beg_idx = 0;
			int max_idx = 0;

			for (int m = beg_idx; m < hyp_cl_amount; ++m) {
				if (max_L_mass[max_idx] < max_L_mass[m])
					max_idx = m;
			}
			flag_summ = 1;
			flag_idx = max_idx;
		}
		delete[] flag_mass;
		delete[] nu_i;
		delete[] interval_bounds;
		delete[] nu_i_bounds;
		delete[] max_L_mass;

		if (flag_summ == 1)
			return flag_idx;
		else return -1;
	};

	auto copy_in_one_mass = [&](double* image_one_mass, int x_c, int y_c, int x_l, int y_l) {
		int idx = 0;
		int i, j;
		for (i = x_c; i < x_c + x_l; ++i) {
			for (j = y_c; j < y_c + y_l; ++j) {
				image_one_mass[idx] = my_picture[i][j];
				idx++;
			}
		}
	};
	
	for (int i = 0; i < iters_i_x+1; ++i) {
		for (int j = 0; j < iters_i_y+1; ++j) {
			if (i ==( iters_i_x ))
				x_l = length_ + (img_l_x - (iters_i_x * half_step + length_));
			else
				x_l = length_;
			if (j == ( iters_i_y ))
				y_l = length_ + (img_l_y - (iters_i_y * half_step + length_));
			else
				y_l = length_;
			copy_in_one_mass(buf_img, i*(half_step), j*(half_step), x_l, y_l);
			
			//idx_class = chi_square_stats(buf_img,  x_l*y_l);
			
			idx_class = kolmogorov_stats2(buf_img, x_l*y_l, mix_shift, mix_scale, hyp_cl_amount );
			//idx_class = max_likehood_stats(buf_img, x_l*y_l, mix_shift, mix_scale, hyp_cl_amount);
			//cout << "idx_class " << idx_class <<"  i*(length_/2), j*(length_/2), "<< i * (half_step)<< " "<< j*(half_step)<< endl;
			if (idx_class != -1)
				for (int l = i * (half_step); l < i*(half_step) +x_l; l++)
					for (int m = j *(half_step); m < j*(half_step) +y_l; m++)
						class_flag[l][m] = idx_class + 1;
		}
	}
	delete[] buf_img;
	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << endl;
}

// раскраска по колмогорову, с распараллеливанием

void mixture_handler::kolmogorov_optimal_redraw_opMP() {
	int length_ = 10;
	auto begin1 = std::chrono::steady_clock::now();
#pragma omp parallel
	{
		int half_step = length_ / 2, j , k, l, m;
		int loc_hyp_cl_amount = hyp_cl_amount;
		int iters_i_x = img_l_x / half_step;
		int iters_i_y = img_l_y / half_step;
		while ((iters_i_x*half_step + length_) > img_l_x)
			iters_i_x--;
		while ((iters_i_y*half_step + length_) > img_l_y)
			iters_i_y--;
		int x_l, y_l, idx_class;
		double* buf_img = new double[4 * length_*length_];

		auto L_max_calculation = [&](double* data, int data_size, int iter, double mix_shift, double mix_scale, double* max_L_mass) {
			double buf_max_l = 0;
			bool flag = false;
			double B;
			for (int m = 0; m < data_size; m++) {
				B = 0;
				if (mix_scale != 0) {
					if (mixture_type == "normal")
						B = (1.0 / (mix_scale))*
						exp(-(pow(data[m] - mix_shift, 2)) /
						(2.0 * mix_scale * mix_scale));
					if (mixture_type == "lognormal")
						B = (1.0 / (mix_scale*data[m]))*
						exp(-(pow(log(data[m]) - mix_shift, 2)) /
						(2.0 * mix_scale * mix_scale));
					if (mixture_type == "rayleigh")
						B = (data[m] / pow(mix_scale, 2))*
						exp(-(pow(data[m], 2)) /
						(2.0 * mix_scale * mix_scale));
				}
				else {
					flag = true;
					break;
				}

				if (B > 0)
					buf_max_l = buf_max_l + log(B / (data_size));
			}

			max_L_mass[iter] = buf_max_l;
		};

		auto kolmogorov_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
			unsigned i, j;
			int intervals_amount = 60;
			int k, buf_intervals_amount;
			bool flag = true;
			double max_d_n, buf_d_n, F_n_curr;
			int* flag_mass = new int[hyp_cl_amount];
			double* stats_mass = new double[hyp_cl_amount];
			int flag_summ = 0;
			int flag_idx = 0;
			double* nu_i = new double[intervals_amount];
			double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
			double min_value = find_k_stat(data, data_size, 0);
			double len_interval = (max_value - min_value) / intervals_amount;
			double *max_L_mass = new double[hyp_cl_amount];
			for (i = 0; i < intervals_amount; ++i)
				nu_i[i] = 0;
			for (i = 0; i < data_size; ++i) {
				k = 0;
				flag = true;
				while (flag) {
					if (((len_interval * k + min_value) <= data[i]) && ((len_interval * (k + 1) + min_value) > data[i]))
						flag = false;
					else
						k++;
				}
				nu_i[k] = nu_i[k] + 1;
			}

			double dn_bound;
			if (length_ == 5)
				dn_bound = 0.264;
			else
				dn_bound = 0.134;
			//double dn_bound = 0.238;
			for (i = 0; i < hyp_cl_amount; ++i) {
				F_n_curr = 0;
				max_d_n = 0;
				if (mixture_type == "normal")
					max_d_n = cdf(normal(mix_shift[i], mix_scale[i]), min_value);
				if (mixture_type == "rayleigh")
					max_d_n = cdf(rayleigh(mix_scale[i]), min_value);
				for (j = 0; j < intervals_amount; ++j) {
					F_n_curr += nu_i[j] / data_size;
					if (j != intervals_amount - 1) {
						for (k = 0; k < 2; ++k) {
							if (mixture_type == "normal")
								buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), (len_interval *(j + k) + min_value)) - F_n_curr);
							if (mixture_type == "rayleigh")
								buf_d_n = abs(cdf(rayleigh(mix_scale[i]), (len_interval *(j + k) + min_value)) - F_n_curr);
							if (buf_d_n > max_d_n)
								max_d_n = buf_d_n;
						}
					}
					else {
						if (mixture_type == "normal")
							buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), (len_interval *(j + 1) + min_value)) - 1);
						if (mixture_type == "rayleigh")
							buf_d_n = abs(cdf(rayleigh(mix_scale[i]), (len_interval *(j + 1) + min_value)) - 1);
						if (buf_d_n > max_d_n)
							max_d_n = buf_d_n;
					}
				}

				stats_mass[i] = max_d_n;
				if (max_d_n < dn_bound)
					flag_mass[i] = 1;
				else
					flag_mass[i] = 0;
			}
			for (i = 0; i < hyp_cl_amount; ++i) {
				flag_summ += flag_mass[i];
				if (flag_mass[i] == 1)
					flag_idx = i;
			}
			if (flag_summ > 1) {
				for (int i = 0; i < hyp_cl_amount; ++i)
					L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

				int beg_idx = 0;
				int max_idx = 0;

				for (int m = beg_idx; m < hyp_cl_amount; ++m) {
					if (max_L_mass[max_idx] < max_L_mass[m])
						max_idx = m;
				}
				flag_summ = 1;
				flag_idx = max_idx;
			}
			delete[] flag_mass;
			delete[] stats_mass;
			delete[] max_L_mass;
			delete[] nu_i;
			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};
		auto max_likehood_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
			unsigned i, j;
			int intervals_amount = 60;
			int k, buf_intervals_amount;
			bool flag = true;
			double max_d_n, buf_d_n, F_n_curr;
			//int* flag_mass = new int[hyp_cl_amount];
			//double* stats_mass = new double[hyp_cl_amount];
			int flag_summ = 0;
			int flag_idx = 0;
			/*double* nu_i = new double[intervals_amount];
			double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
			double min_value = find_k_stat(data, data_size, 0);
			double len_interval = (max_value - min_value) / intervals_amount;*/
			double *max_L_mass = new double[hyp_cl_amount];


			for (int i = 0; i < hyp_cl_amount; ++i)
				L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

			int beg_idx = 0;
			int max_idx = 0;

			for (int m = beg_idx; m < hyp_cl_amount; ++m) {
				if (max_L_mass[max_idx] < max_L_mass[m])
					max_idx = m;
			}
			flag_summ = 1;
			flag_idx = max_idx;


			delete[] max_L_mass;

			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};
		auto kolmogorov_stats2 = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
			unsigned i, j;
			//int intervals_amount = 60;
			int k, buf_intervals_amount;
			bool flag = true;
			double max_d_n, buf_d_n, F_n_curr;
			int* flag_mass = new int[hyp_cl_amount];
			double* stats_mass = new double[hyp_cl_amount];
			int flag_summ = 0;
			int flag_idx = 0;

			/*double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
			double min_value = find_k_stat(data, data_size, 0);*/
			//double len_interval = (max_value - min_value) / intervals_amount;
			double *max_L_mass = new double[hyp_cl_amount];



			//double temp;
			//for (int i = 0; i < data_size - 1; i++) {
			//	for (int j = 0; j < data_size - i - 1; j++) {
			//		if (data[j] > data[j + 1]) {
			//			// меняем элементы местами
			//			temp = data[j];
			//			data[j] = data[j + 1];
			//			data[j + 1] = temp;
			//		}
			//	}
			//}
//			
//#pragma omp critical
//			{
//				cout << "quickSort!" << endl;
//				quickSort(data, 0, data_size-1, data_size / 2);
//				for (int i = 0; i < data_size; ++i)
//					cout << data[i] << endl;
//			}
			quickSort(data, 0, data_size - 1, data_size / 2);
			double dn_bound;
			if (length_ == 5)
				dn_bound = 0.264;
			else
				dn_bound = 0.134;
			//double dn_bound = 0.238;
			for (i = 0; i < hyp_cl_amount; ++i) {
				F_n_curr = 0;
				max_d_n = 0;
				if (mixture_type == "lognormal")
					max_d_n = cdf(lognormal(mix_shift[i], mix_scale[i]), data[0]);
				else {
					if (mixture_type == "normal")
						max_d_n = cdf(normal(mix_shift[i], mix_scale[i]), data[0]);
					else
						if (mixture_type == "rayleigh")
							max_d_n = cdf(rayleigh(mix_scale[i]), data[0]);
				}
				
				for (j = 1; j < data_size; ++j) {
					F_n_curr += 1.0 / data_size;
					if (j != data_size - 1) {
						for (k = 0; k < 2; ++k) {
							if (mixture_type == "lognormal")
								buf_d_n = abs(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]) - (F_n_curr - k * 1.0 / data_size));
							else {
								if (mixture_type == "normal")
									buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - (F_n_curr - k * 1.0 / data_size));
								else {
									if (mixture_type == "rayleigh")
										buf_d_n = abs(cdf(rayleigh(mix_scale[i]), data[j]) - (F_n_curr - k * 1.0 / data_size));
								}
							}
						}
					}
					else {
						if (mixture_type == "lognormal")
							buf_d_n = abs(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]) - 1);
						else {
							if (mixture_type == "normal")
								buf_d_n = abs(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - 1);
							else {
								if (mixture_type == "rayleigh")
									buf_d_n = abs(cdf(rayleigh(mix_scale[i]), data[j]) - 1);
							}
						}	
					}
					if (buf_d_n > max_d_n)
						max_d_n = buf_d_n;
				}
				
				stats_mass[i] = max_d_n;
				if (max_d_n < dn_bound)
					flag_mass[i] = 1;
				else
					flag_mass[i] = 0;
			}
			for (i = 0; i < hyp_cl_amount; ++i) {
				flag_summ += flag_mass[i];
				if (flag_mass[i] == 1)
					flag_idx = i;
			}
			if (flag_summ > 1) {
				for (int i = 0; i < hyp_cl_amount; ++i)
					L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

				int beg_idx = 0;
				int max_idx = 0;

				for (int m = beg_idx; m < hyp_cl_amount; ++m) {
					if (max_L_mass[max_idx] < max_L_mass[m])
						max_idx = m;
				}
				flag_summ = 1;
				flag_idx = max_idx;
			}
			delete[] flag_mass;
			delete[] stats_mass;
			delete[] max_L_mass;

			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};
		auto chi_square_stats = [&](double* data, int data_size) {
			int i, j, k, mix_params_amount, buf_intervals_amount;
			double chi_stat, teor_nu, quant_chi;
			bool flag = true;
			int intervals_amount = 30;
			double max_value = find_k_stat(data, data_size, data_size - 1) + 1;
			double min_value = find_k_stat(data, data_size, 0) - 1;
			double len_interval = (max_value - min_value) / intervals_amount;
			if (mixture_type == "normal")
				mix_params_amount = 2;
			if (mixture_type == "rayleigh")
				mix_params_amount = 1;
			if (mixture_type == "lognormal")
				mix_params_amount = 2;
			double* nu_i = new double[intervals_amount];
			double* interval_bounds = new double[intervals_amount];
			double* nu_i_bounds = new double[intervals_amount];
			double *max_L_mass = new double[hyp_cl_amount];
			int* flag_mass = new int[hyp_cl_amount];

			for (i = 0; i < intervals_amount; ++i)
				nu_i[i] = 0;
			for (i = 0; i < data_size; ++i) {
				k = 0;
				flag = true;
				while (flag) {
					if (((len_interval * k + min_value) <= data[i]) && ((len_interval * (k + 1) + min_value) > data[i]))
						flag = false;
					else
						k++;
				}
				nu_i[k] = nu_i[k] + 1;
			}
			flag = true;
			int r = data_size;
			int buf_nu = 0;
			buf_intervals_amount = 0;
			for (int j = 0; j < intervals_amount; ++j) {
				//cout << "nu_i[k] " << nu_i[j] << endl;
				if ((nu_i[j] > 5) && flag) {
					if (j < intervals_amount - 1) {
						if ((r - nu_i[j] > 5)) {
							interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
							nu_i_bounds[buf_intervals_amount] = nu_i[j];
							r -= nu_i[j];
							buf_intervals_amount++;
						}
						else {
							flag = false;
							buf_nu = nu_i[j];
							r -= nu_i[j];
						}
					}
					else {
						interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
						nu_i_bounds[buf_intervals_amount] = nu_i[j];
						r -= nu_i[j];
						buf_intervals_amount++;
					}
				}
				else {
					if (flag) {
						buf_nu = nu_i[j];
						r -= nu_i[j];
						flag = false;
					}
					else {
						buf_nu += nu_i[j];
						r -= nu_i[j];
					}
					if (buf_nu > 5) {
						if (j < intervals_amount - 1) {
							if (r > 5)
							{
								interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
								nu_i_bounds[buf_intervals_amount] = buf_nu;
								buf_intervals_amount++;
								buf_nu = 0;

								flag = true;
							}
						}
						else {
							//cout << "j " << j << " buf_nu " << buf_nu << endl;
							interval_bounds[buf_intervals_amount] = len_interval * (j + 1) + min_value;
							nu_i_bounds[buf_intervals_amount] = buf_nu;
							buf_intervals_amount++;
							buf_nu = 0;

							flag = true;
						}
					}
				}

			}

			int flag_summ = 0;
			int flag_idx = 0;

			if (buf_intervals_amount > mix_params_amount + 1) {
				//cout << "+" << endl;
				for (i = 0; i < hyp_cl_amount; ++i) {
					chi_stat = 0;

					for (j = 0; j < buf_intervals_amount; ++j) {
						if (j == 0) {
							if (mixture_type == "normal")
								teor_nu = cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j]);
							if (mixture_type == "lognormal")
								teor_nu = cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j]);
							if (mixture_type == "rayleigh")
								teor_nu = cdf(rayleigh(mix_scale[i]), interval_bounds[j]);
						}
						else {
							if (j != buf_intervals_amount - 1) {
								if (mixture_type == "normal")
									teor_nu = cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j])
									- cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
								if (mixture_type == "lognormal")
									teor_nu = cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j])
									- cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
								if (mixture_type == "rayleigh")
									teor_nu = cdf(rayleigh(mix_scale[i]), interval_bounds[j])
									- cdf(rayleigh(mix_scale[i]), interval_bounds[j - 1]);
							}
							else {
								if (mixture_type == "normal")
									teor_nu = 1 - cdf(normal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
								if (mixture_type == "lognormal")
									teor_nu = 1 - cdf(lognormal(mix_shift[i], mix_scale[i]), interval_bounds[j - 1]);
								if (mixture_type == "rayleigh")
									teor_nu = 1 - cdf(rayleigh(mix_scale[i]), interval_bounds[j - 1]);
							}
						}
						//cout << "teor_nu " << teor_nu << endl;
						teor_nu = teor_nu * (data_size);
						chi_stat += (nu_i_bounds[j] - teor_nu)* (nu_i_bounds[j] - teor_nu) / teor_nu;

					}
					quant_chi = quantile(chi_squared(buf_intervals_amount - 1 - mix_params_amount), 0.99);
					//cout << "classs " << i << ": chi_stat - " << chi_stat << " quant_chi = " << quant_chi << endl;
					if (chi_stat < quant_chi)
						flag_mass[i] = 1;
					else
						flag_mass[i] = 0;
				}

				for (i = 0; i < hyp_cl_amount; ++i) {
					flag_summ += flag_mass[i];
					if (flag_mass[i] == 1)
						flag_idx = i;
				}
			}
			else
				flag_summ = 0;

			if (flag_summ > 1) {
				for (int i = 0; i < hyp_cl_amount; ++i)
					L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

				int beg_idx = 0;
				int max_idx = 0;

				for (int m = beg_idx; m < hyp_cl_amount; ++m) {
					if (max_L_mass[max_idx] < max_L_mass[m])
						max_idx = m;
				}
				flag_summ = 1;
				flag_idx = max_idx;
			}
			delete[] flag_mass;
			delete[] nu_i;
			delete[] interval_bounds;
			delete[] nu_i_bounds;
			delete[] max_L_mass;

			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};
		auto kramer_mizes_smirnoff_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
			unsigned i, j;
			//int intervals_amount = 60;
			int k, buf_intervals_amount;
			bool flag = true;
			double max_d_n, buf_d_n, F_n_curr;
			int* flag_mass = new int[hyp_cl_amount];
			double* stats_mass = new double[hyp_cl_amount];
			double* Ui_mass = new double[data_size];
			int flag_summ = 0;
			int flag_idx = 0;
			int unic_amount = 1;
			double last_elem;
			double *max_L_mass = new double[hyp_cl_amount];
			quickSort(data, 0, data_size - 1, data_size / 2);
			/*last_elem  = data[0];
			Ui_mass[0] = data[0];
			for (int i = 1; i < data_size; ++i) {
				if (data[i] != last_elem) {
					Ui_mass[unic_amount] = data[i];
					
					unic_amount++;
				}
				last_elem = data[i];
			}*/
//#pragma omp critical
//			{
//				cout << "mass - " << endl;
//				for (j = 0; j < data_size; ++j) {
//					cout << data[j] << endl;
//				}
//				cout << "line - " << unic_amount << endl;
//				for (j = 0; j < unic_amount; ++j) {
//					cout << Ui_mass[j] << endl;
//				}
//			}
			double dn_bound = 0.4614;
			//double dn_bound = 0.7484;
			//double dn_bound = 0.45778;
			//double dn_bound = 0.45986;
			
			for (i = 0; i < hyp_cl_amount; ++i) {
				F_n_curr = 0;
				buf_d_n = 0;
				/*for (j = 0; j < unic_amount; ++j) {
					if (mixture_type == "lognormal")
						buf_d_n += pow(cdf(lognormal(mix_shift[i], mix_scale[i]), Ui_mass[j]) - (2*(j+1)-1 )/ double(2*unic_amount),2);
					else {
						if (mixture_type == "normal")
							buf_d_n += pow(cdf(normal(mix_shift[i], mix_scale[i]), Ui_mass[j]) - (2 * (j + 1) - 1) / double(2 * unic_amount), 2);
						else {
							if (mixture_type == "rayleigh")
								buf_d_n += pow(cdf(rayleigh(mix_scale[i]), Ui_mass[j]) - (2 * (j + 1) - 1) / double(2 * unic_amount), 2);
						}
					}
				}
				buf_d_n += 1.0 / double(12 * unic_amount);*/
				for (j = 0; j < data_size; ++j) {
					if (mixture_type == "lognormal")
						buf_d_n += pow(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
					else {
						if (mixture_type == "normal")
							buf_d_n += pow(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
						else {
							if (mixture_type == "rayleigh")
								buf_d_n += pow(cdf(rayleigh(mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
						}
					}
				}
				buf_d_n += 1.0 / double(12 * data_size);
				buf_d_n = (buf_d_n  - 0.4 / double(data_size) + 0.6 / double(data_size*data_size))*(1 + 1.0 / double(data_size));
				//buf_d_n = (buf_d_n - 0.03 / double(data_size) )*(1 + 0.5 / double(data_size));
				//cout << "buf_d_n - " << buf_d_n << endl;
				if (buf_d_n < dn_bound)
					flag_mass[i] = 1;
				else
					flag_mass[i] = 0;
			}
			for (i = 0; i < hyp_cl_amount; ++i) {
				flag_summ += flag_mass[i];
				if (flag_mass[i] == 1)
					flag_idx = i;
			}
			if (flag_summ > 1) {
				for (int i = 0; i < hyp_cl_amount; ++i)
					L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

				int beg_idx = 0;
				int max_idx = 0;

				for (int m = beg_idx; m < hyp_cl_amount; ++m) {
					if (max_L_mass[max_idx] < max_L_mass[m])
						max_idx = m;
				}
				flag_summ = 1;
				flag_idx = max_idx;
			}
			delete[] flag_mass;
			delete[] stats_mass;
			delete[] max_L_mass;
			delete[] Ui_mass;
			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};
		auto watson_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
			unsigned i, j;
			//int intervals_amount = 60;
			int k, buf_intervals_amount;
			bool flag = true;
			double max_d_n, buf_d_n, F_n_curr;
			int* flag_mass = new int[hyp_cl_amount];
			double* stats_mass = new double[hyp_cl_amount];
			double* Ui_mass = new double[data_size];
			int flag_summ = 0;
			int flag_idx = 0;
			int unic_amount = 1;
			double last_elem;
			double *max_L_mass = new double[hyp_cl_amount];
			quickSort(data, 0, data_size - 1, data_size / 2);
			
			double dn_bound = 0.186880;

			for (i = 0; i < hyp_cl_amount; ++i) {
				F_n_curr = 0;
				buf_d_n = 0;
				
				for (j = 0; j < data_size; ++j) {
					if (mixture_type == "lognormal") {
						buf_d_n += pow(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]) - ( (j + 1) - 0.5) / double( data_size), 2);
						F_n_curr += cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]);
					}
					else {
						if (mixture_type == "normal")
							buf_d_n += pow(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
						else {
							if (mixture_type == "rayleigh")
								buf_d_n += pow(cdf(rayleigh(mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
						}
					}
				}
				buf_d_n += 1.0 / double(12 * data_size)- data_size*pow(F_n_curr/double(data_size)-0.5,2);
				//buf_d_n = (buf_d_n / 4.0 - 0.4 / data_size + 0.6 / double(data_size*data_size))*(1 + 1.0 / double(data_size));
				//cout << "buf_d_n - " << buf_d_n << endl;
				if (buf_d_n < dn_bound)
					flag_mass[i] = 1;
				else
					flag_mass[i] = 0;
			}
			for (i = 0; i < hyp_cl_amount; ++i) {
				flag_summ += flag_mass[i];
				if (flag_mass[i] == 1)
					flag_idx = i;
			}
			if (flag_summ > 1) {
				for (int i = 0; i < hyp_cl_amount; ++i)
					L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

				int beg_idx = 0;
				int max_idx = 0;

				for (int m = beg_idx; m < hyp_cl_amount; ++m) {
					if (max_L_mass[max_idx] < max_L_mass[m])
						max_idx = m;
				}
				flag_summ = 1;
				flag_idx = max_idx;
			}
			delete[] flag_mass;
			delete[] stats_mass;
			delete[] max_L_mass;
			delete[] Ui_mass;
			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};

		auto anderson_stats = [&](double* data, int data_size, double* mix_shift, double* mix_scale, int  hyp_cl_amount) {
			int i, j;
			//int intervals_amount = 60;
			int k, buf_intervals_amount;
			bool flag = true;
			double max_d_n, buf_d_n, F_n_curr;
			int* flag_mass = new int[hyp_cl_amount];
			double* stats_mass = new double[hyp_cl_amount];
			double* Ui_mass = new double[data_size];
			int flag_summ = 0;
			int flag_idx = 0;
			int unic_amount = 1;
			double last_elem;
			double *max_L_mass = new double[hyp_cl_amount];
			quickSort(data, 0, data_size - 1, data_size / 2);

			double dn_bound = 2.4924;

			for (i = 0; i < hyp_cl_amount; ++i) {
				F_n_curr = 0;
				buf_d_n = 0;

				for (j = 0; j < data_size; ++j) {
					if (mixture_type == "lognormal") {
						buf_d_n += log(cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]))*((2*(j+1)-1)/(double(2*data_size)))
							+ (-(2*(j + 1)-1) / double(2*data_size)+1)*log(1.0-cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]));
						//F_n_curr += cdf(lognormal(mix_shift[i], mix_scale[i]), data[j]);
					}
					else {
						if (mixture_type == "normal")
							buf_d_n += pow(cdf(normal(mix_shift[i], mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
						else {
							if (mixture_type == "rayleigh")
								buf_d_n += pow(cdf(rayleigh(mix_scale[i]), data[j]) - (2 * (j + 1) - 1) / double(2 * data_size), 2);
						}
					}
				}
				buf_d_n = -buf_d_n*2- data_size;
				//buf_d_n = (buf_d_n / 4.0 - 0.4 / data_size + 0.6 / double(data_size*data_size))*(1 + 1.0 / double(data_size));
				//cout << "buf_d_n - " << buf_d_n << endl;
				if (buf_d_n < dn_bound)
					flag_mass[i] = 1;
				else
					flag_mass[i] = 0;
			}
			for (i = 0; i < hyp_cl_amount; ++i) {
				flag_summ += flag_mass[i];
				if (flag_mass[i] == 1)
					flag_idx = i;
			}
			if (flag_summ > 1) {
				for (int i = 0; i < hyp_cl_amount; ++i)
					L_max_calculation(data, data_size, i, mix_shift[i], mix_scale[i], max_L_mass);

				int beg_idx = 0;
				int max_idx = 0;

				for (int m = beg_idx; m < hyp_cl_amount; ++m) {
					if (max_L_mass[max_idx] < max_L_mass[m])
						max_idx = m;
				}
				flag_summ = 1;
				flag_idx = max_idx;
			}
			delete[] flag_mass;
			delete[] stats_mass;
			delete[] max_L_mass;
			delete[] Ui_mass;
			if (flag_summ == 1)
				return flag_idx;
			else return -1;
		};
		auto copy_in_one_mass = [&](double* image_one_mass, int x_c, int y_c, int x_l, int y_l) {
			int idx = 0;
			int i, j;
			for (i = x_c; i < x_c + x_l; ++i) {
				for (j = y_c; j < y_c + y_l; ++j) {
					image_one_mass[idx] = my_picture[i][j];
					idx++;
				}
			}
		};
		x_l = length_;
		#pragma omp for
		for (int i = 0; i < iters_i_x + 1; ++i) {
			if (i == iters_i_x)
				x_l = length_ + (img_l_x - (iters_i_x * half_step + length_));
			y_l = length_;
			for ( int j = 0; j < iters_i_y + 1; ++j) {
				
				if (j == iters_i_y)
					y_l = length_ + (img_l_y - (iters_i_y * half_step + length_));
				
				copy_in_one_mass(buf_img, i*(half_step), j*(half_step), x_l, y_l);

				//idx_class = chi_square_stats(buf_img,  x_l*y_l);

				
				idx_class = kolmogorov_stats2(buf_img, x_l*y_l, mix_shift, mix_scale, loc_hyp_cl_amount);
				//idx_class = kramer_mizes_smirnoff_stats(buf_img, x_l*y_l, mix_shift, mix_scale, loc_hyp_cl_amount);
				//idx_class = watson_stats(buf_img, x_l*y_l, mix_shift, mix_scale, loc_hyp_cl_amount);
				//idx_class = anderson_stats(buf_img, x_l*y_l, mix_shift, mix_scale, loc_hyp_cl_amount);
				//idx_class = max_likehood_stats(buf_img, x_l*y_l, mix_shift, mix_scale, hyp_cl_amount);
				//cout << "idx_class " << idx_class <<"  i*(length_/2), j*(length_/2), "<< i * (half_step)<< " "<< j*(half_step)<< endl;
				if (idx_class != -1) {
					#pragma omp critical
					{
						for (l = i * (half_step); l < i*(half_step)+x_l; ++l)
							for (m = j * (half_step); m < j*(half_step)+y_l; ++m)
								class_flag[l][m] = idx_class + 1;
					}
				}
			}
		}
		delete[] buf_img;
	}
	auto end1 = std::chrono::steady_clock::now();
	auto elapsed_ms1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1);
	cout << "elapsed_ms1  " << elapsed_ms1.count() << endl;
}

//BIC

void mixture_handler::BIC() {
	double summ = 0;
	double big_summ = 0;
	long double f_i_j, pix_buf;
	unsigned i, j, idx_i, idx_j;
	

	for (i = 0; i < img_l_x*img_l_y; ++i) {
		idx_i = i / img_l_y;
		idx_j = i % img_l_y;
		summ = 0;
		pix_buf = my_picture[idx_i][idx_j];
		for (j = 0; j < hyp_cl_amount; ++j)
			summ += mix_weight[j] * (1 / (mix_scale[j] * sqrt(2 * pi)))*exp(-((pix_buf
				- mix_shift[j])*(pix_buf - mix_shift[j])) / (2.0 * mix_scale[j] * mix_scale[j]));

		big_summ += log(summ);
	}
	unsigned count_n_z = 0;
	/*for (j = 0; j < hyp_cl_amount; ++j)
		if (mix_weight[j] != 0)
			++count_n_z;*/
	count_n_z = hyp_cl_amount;
	bic_value = -2 * big_summ + log(img_l_x*img_l_y)*(3 * count_n_z - 1);

	cout << "BIC:  " << bic_value << "     " << big_summ << "\n" << "\n" << endl;
	
}

// загрузка результат классификации в файл 

void mixture_handler::printInformation_to_image() {
	CImage result;
	result.Create(img_l_y, img_l_x, 24);
	int cl_idx_color_g = 0;
	int cl_idx_color_r = 0;
	int cl_idx_color_b = 0;
	// задаем цвет пикселя
	for (int i = 0; i < img_l_x; i++) {
		for (int j = 0; j < img_l_y; j++) {
			cl_idx_color_g = (255/hyp_cl_amount)* class_flag[img_l_x - i - 1][j];
			cl_idx_color_b = (255 / (hyp_cl_amount/2))*(class_flag[img_l_x - i - 1][j] % 3);
			cl_idx_color_r = (255 / (hyp_cl_amount / 3))*(class_flag[img_l_x - i - 1][j] % 3);
			result.SetPixelRGB(j, i, cl_idx_color_r, cl_idx_color_g, cl_idx_color_b);
		}
	}

	CString _name = L"D:\\classification_image_" + gen_image->get_item_type() + L".jpg";
	/*LPCTSTR file_name = LPCTSTR(_name.c_str());
	cout << file_name << endl;*/
	result.Save(_name);

}

void mixture_handler::detect_result_by_mask() {
	CImage mask_image;
	double amount_cl_pix = 0;
	unsigned curr_class;
	double amount_true_pix = 0;
	int y_len, x_len, i, j, k;
	cout << "percentage of correctly classified pixels:" << endl;
	for (k = 0; k < hyp_cl_amount; ++k) {
		if (gen_image->get_mask_list()[k] != L"")
		{
			mask_image.Load(gen_image->get_mask_list()[k]);
			amount_cl_pix = 0;
			amount_true_pix = 0;
			for (i = 0; i < img_l_x; i++) {
				for (j = 0; j < img_l_y; j++) {
					curr_class = (unsigned(GetBValue(mask_image.GetPixel(j, i))) / 255)*(k + 1);
					if (curr_class != 0) {
						amount_cl_pix++;
						if (class_flag[img_l_x - i - 1][j] == curr_class)
							amount_true_pix++;
					}

				}
			}
			mask_image.Detach();
			cout << "class " << k + 1 << ", "<< curr_class<< ": " << amount_true_pix / amount_cl_pix << endl;

		}
		
	}
}

// вывод информации о полученном результате классификации

void mixture_handler::printInformation() {
	unsigned i, j;
	cout << "finded model:" << "\n";
	cout << " mixture components: " << hyp_cl_amount << "\n";
	cout << "EM mix_shift values:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_shift[i] << "  ";
	cout << endl;
	cout << endl;
	cout << "EM mix_scale values:" << "\n";
	for (i = 0; i < hyp_cl_amount; i++)
		cout << mix_scale[i] << "  ";
	cout << endl;

	out.open(split_mix_filename);
	for (i = 0; i < img_l_x; i++) {
		for (j = 0; j < img_l_y; j++)
			out << class_flag[i][j] << " ";
		out << std::endl;
	}
	out.close();
}

//поиск медианы

double mixture_handler::find_med(double* window, int wind_size) {
	int med_index = (wind_size) / 2 - 1;
	bool flag = true;
	int left = 0;
	int right = wind_size - 1;
	if (med_index >= 0) {
		while (flag) {
			//cout << "ddd";
			std::pair<int, int> result = partition(window, left, right, med_index);
			if (result.first< med_index && result.second > med_index) {
				flag = false;

			}
			else {
				if (result.first > med_index)
					right = result.first;
				else {
					if (result.second < med_index)
						left = result.second;
				}
			}
		}

		return 	window[med_index];
	}
	else
		return -1;
}

//поиск к-той порядковой статистики

double mixture_handler::find_k_stat(double * data, int wind_size, int k_stat) {
	bool flag = true;
	int  left = 0;
	int  right = wind_size - 1;

	while (flag) {
		std::pair<int, int> result = partition(data, left, right, k_stat);
		if (result.first< k_stat && result.second > k_stat)
			flag = false;
		else {
			if (result.first > k_stat)
				right = result.first;
			else {
				if (result.second < k_stat)
					left = result.second;
			}
		}
	}

	return 	data[k_stat];
}

// быстрая сортировка

void  mixture_handler::quickSort(double * data,  int l, int r, int pivot_index) {
	double v , temp;
	int i ,	j ,	p ,	q ;
	while (l < r) {
		v = data[r];
		i = l;
		j = r - 1;
		p = l - 1;
		q = r;
		while (i <= j) {
			while (data[i] < v)
				i++;
			while (data[j] > v)
				j--;
			if (i >= j)
				break;

			temp = data[i];
			data[i] = data[j];
			data[j] =temp;
			//swap(data[i], data[j]);
			if (data[i] == v) {
				p++;
				temp = data[p];
				data[p] = data[i];
				data[i] = temp;
				//swap(data[i], data[p]);
			}
			i++;
			if (data[j] == v) {
				q--;
				temp = data[q];
				data[q] = data[j];
				data[j] = temp;
				//swap(data[q], data[j]);
			}
			j--;
		}
		temp = data[i];
		data[i] = data[r];
		data[r] = temp;
		//swap(data[i], data[r]);
		j = i - 1;
		i++;
		for (int k = l; k <= p; k++, j--) {
			temp = data[k];
			data[k] = data[j];
			data[j] = temp;
			//swap(data[k], data[j]);
		}

		for (int k = r - 1; k >= q; k--, i++) {
			temp = data[k];
			data[k] = data[i];
			data[i] = temp;
			//swap(data[i], data[k]);
		}

		if ((j - l )<( r - i)){
			quickSort(data, l, j, 0);
			l = i;
		}
		else {
			quickSort(data, i, r, 0);
			r = j;
		}
	}
}

//partition

//std::pair<int, int> mixture_handler::partition(double* mass, int left, int right, int  ind_pivot) {
//	double pivot = mass[ind_pivot];
//	double buf = mass[left];
//	mass[left] = pivot;
//	mass[ind_pivot] = buf;
//	int j = left;
//	int	k = right + 1;
//	int	iter_l = left + 1;
//	int	iter_r = right;
//
//	while (iter_l <= iter_r) {
//		while ((iter_l <= iter_r) && (mass[iter_l] < mass[left])) {
//if (j == iter_l - 1) {
//	j++;
//	iter_l++;
//}
//else {
//	j++;
//	buf = mass[iter_l];
//	mass[iter_l] = mass[j];
//	mass[j] = buf;
//	iter_l++;
//}
//		}
//		while ((iter_l <= iter_r) && (mass[iter_r] > mass[left]) && (iter_r >= left)) {
//			if (k == iter_r + 1) {
//				k -= 1;
//				iter_r -= 1;
//			}
//			else {
//				k -= 1;
//				buf = mass[iter_r];
//				mass[iter_r] = mass[k];
//				mass[k] = buf;
//				iter_r -= 1;
//			}
//		}
//		if (iter_l <= iter_r) {
//			if ((mass[iter_l] != mass[left]) && (mass[iter_r] != mass[left]) && (iter_r >= left)) {
//				buf = mass[iter_r];
//				mass[iter_r] = mass[iter_l];
//				mass[iter_l] = buf;
//				j++;
//				buf = mass[iter_l];
//				mass[iter_l] = mass[j];
//				mass[j] = buf;
//				iter_l++;
//				k -= 1;
//				buf = mass[iter_r];
//				mass[iter_r] = mass[k];
//				mass[k] = buf;
//				iter_r -= 1;
//			}
//			else {
//				if ((mass[iter_l] == mass[left]) && (mass[iter_r] != mass[left]) && (iter_r >= left)) {
//					buf = mass[iter_r];
//					mass[iter_r] = mass[iter_l];
//					mass[iter_l] = buf;
//					j++;
//					buf = mass[j];
//					mass[j] = mass[iter_l];
//					mass[iter_l] = buf;
//					iter_l++;
//				}
//				else {
//					if ((mass[iter_l] != mass[left]) && (mass[iter_r] == mass[left]) && (iter_r >= left)) {
//						buf = mass[iter_r];
//						mass[iter_r] = mass[iter_l];
//						mass[iter_l] = buf;
//						k -= 1;
//						buf = mass[iter_r];
//						mass[iter_r] = mass[k];
//						mass[k] = buf;
//						iter_r -= 1;
//					}
//					else {
//						if ((mass[iter_l] == mass[left]) && (mass[iter_r] == mass[left]) && (iter_r >= left)) {
//							iter_r -= 1;
//							while ((iter_l <= iter_r) && (mass[iter_r] == mass[left]))
//								iter_r -= 1;
//							if (iter_l < iter_r) {
//								if (mass[iter_r] < mass[left]) {
//									buf = mass[iter_r];
//									mass[iter_r] = mass[j + 1];
//									mass[j + 1] = buf;
//									j++;
//									iter_l++;
//								}
//								else {
//									buf = mass[iter_r];
//									mass[iter_r] = mass[k - 1];
//									mass[k - 1] = buf;
//									k -= 1;
//									iter_r -= 1;
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//
//	buf = mass[left];
//	mass[left] = mass[j];
//	mass[j] = buf;
//
//	return std::pair<int, int>(j - 1, k);
//}

std::pair<int, int> mixture_handler::partition(double* mass, int left, int right, int  ind_pivot)
{
	double v = mass[ind_pivot];
	//	double buf = mass[left];
	//double v = mass[right];

	double temp = mass[right];
	mass[right] = v;
	mass[ind_pivot] = temp;
	int i = left;
	int j = right - 1;
	int p = left - 1;
	int q = right;
	while (i <= j) {
		while (mass[i] < v)
			i++;
		while (mass[j] > v)
			j--;
		if (i >= j)
			break;
		
		temp = mass[i];
		mass[i] = mass[j];
		mass[j] = temp;
		if (mass[i] == v) {
			p++;
			temp = mass[p];
			mass[p] = mass[i];
			mass[i] = temp;
		}
		i++;
		if (mass[j] == v) {
			q--;
			temp = mass[q];
			mass[q] = mass[j];
			mass[j] = temp;
		}
		j--;
	}
	temp = mass[i];
	mass[i] = mass[right];
	mass[right] = temp;
	j = i - 1;
	i++;
	for (int k = left; k <= p; k++, j--) {
		temp = mass[k];
		mass[k] = mass[j];
		mass[j] = temp;
	}
		
	for (int k = right - 1; k >= q; k--, i++) {
		temp = mass[k];
		mass[k] = mass[i];
		mass[i] = temp;
	}
		
	return  std::pair<int, int>(j , i);

}

//оценка правильности результатов

void mixture_handler::detect_results() {
	mistake_mix = new float[hyp_cl_amount];
	float r_far_ = 0;

	for (int i = 0; i < hyp_cl_amount; i++)
		mistake_mix[i] = 0;
	thr_nmb = 5;
	rfar = new float[thr_nmb];
	for (int i = 0; i < thr_nmb; ++i)
		rfar[i] = 0.0;
	std::thread threadObj1(&mixture_handler::th_detect_results, this, 0, img_l_x / thr_nmb, 0);
	std::thread threadObj2(&mixture_handler::th_detect_results, this, img_l_x / thr_nmb, 2 * img_l_x / thr_nmb, 1);
	std::thread threadObj3(&mixture_handler::th_detect_results, this, 2 * img_l_x / thr_nmb, 3 * img_l_x / thr_nmb, 2);
	std::thread threadObj4(&mixture_handler::th_detect_results, this, 3 * img_l_x / thr_nmb, 4 * img_l_x / thr_nmb, 3);
	std::thread threadObj5(&mixture_handler::th_detect_results, this, 4 * img_l_x / thr_nmb, img_l_x, 4);
	threadObj1.join();
	threadObj2.join();
	threadObj3.join();
	threadObj4.join();
	threadObj5.join();

	int trg_pix = 0;
	for (int i = 0; i < 25; i++)
		trg_pix += gen_image->get_targets()[i].size*gen_image->get_targets()[i].size;
	for (int i = 0; i < thr_nmb; ++i)
		r_far_ += rfar[i];
	cout << "trg_pix" << trg_pix / 5 << endl;
	cout << "real far: " << r_far_ / (img_l_x*img_l_x - trg_pix) << endl;
	cout << "all_mistakes: " << all_mistakes << endl;
	cout << "mistakes for each mixture: ";
	for (int i = 0; i < hyp_cl_amount; i++) {
		cout << mistake_mix[i] / all_mistakes << "  ";
	}
	cout << endl;
	delete[] mistake_mix;
}

//обработка результатов - поточна версия

void mixture_handler::th_detect_results(int beg, int end, int thr) {
	int l_numb = 0;
	float n_hit = 0.0;
	float n_miss = 0.0;
	float n_detect = 0.0;
	double x_min, x_max, y_min, y_max;
	int   n_block = 0;
	unsigned comp_numb = 0;
	double min_dist = sqrt((mix_shift[0] - gen_image->get_bask_shift())*(mix_shift[0] - gen_image->get_bask_shift())
		+ (mix_scale[0] - gen_image->get_bask_scale())*(mix_scale[0] - gen_image->get_bask_scale()));
	double buf_dist;
	for (int i = 0; i < hyp_cl_amount; ++i) {
		buf_dist = sqrt((mix_shift[i] - gen_image->get_bask_shift())*(mix_shift[i] - gen_image->get_bask_shift())
			+ (mix_scale[i] - gen_image->get_bask_scale())*(mix_scale[i] - gen_image->get_bask_scale()));
		if (min_dist >= buf_dist) {
			min_dist = buf_dist;
			comp_numb = i + 1;
		}
	}
	for (int i = 0; i < 5; i++) {
		n_hit = 0.0;
		n_miss = 0.0;
		n_detect = 0.0;

		for (int j = beg; j < end; j++) {
			n_block = 0;
			while (j < n_block + 1 && j >= n_block)
				n_block++;
			n_block = int(j / 110);
			for (int k = i * 110; k < (i + 1) * 110; k++) {
				x_min = gen_image->get_targets()[n_block * 5 + i].x;
				x_max = x_min + gen_image->get_targets()[n_block * 5 + i].size;
				y_min = gen_image->get_targets()[n_block * 5 + i].y;
				y_max = y_min + gen_image->get_targets()[n_block * 5 + i].size;

				if (j < x_max &&j >= x_min) {
					if (k < y_max &&k >= y_min) {
						if (class_flag[j][k] != gen_image->get_targets()[n_block * 5 + i].mix_type) {
							all_mistakes++;
							mistake_mix[class_flag[j][k] - 1] ++;
						}
					}
					else {
						if (class_flag[j][k] != comp_numb) {
							rfar[thr] += 1;
							all_mistakes++;
							mistake_mix[class_flag[j][k] - 1] ++;
						}

					}

				}
				else {

					if (class_flag[j][k] != comp_numb) {
						rfar[thr] += 1;
						all_mistakes++;
						mistake_mix[class_flag[j][k] - 1] ++;
					}

				}
			}
		}
	}
}

// вызов скрипта на python для отрисовки результатов

void mixture_handler::draw_graphics() {
	string cmd = "echo python  C:\\Users\\anastasya\\PycharmProjects\\untitled5\\mixture_vizualization.py " + gen_image->get_filename() + " " + split_mix_filename +
		" | %windir%\\system32\\cmd.exe \"/K\" C:\\Users\\anastasya\\Anaconda3\\Scripts\\activate.bat  ";
	system(cmd.c_str());
}

// деструктор

mixture_handler::~mixture_handler()
{
	for (int i = 0; i < img_l_x; i++)
		delete[] class_flag[i];

	delete[] class_flag;
	delete[] mix_shift;
	delete[] mix_scale;
	delete[] mix_weight;
	delete[] mix_prob;
	delete[] rfar;
}
