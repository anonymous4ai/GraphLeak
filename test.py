import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random

# 设置
pd.options.display.notebook_repr_html = False  # 表格显示
plt.rcParams["figure.dpi"] = 400  # 图形分辨率
sns.set_theme(style="darkgrid")  # 图形主题
sns.set_context("poster", font_scale=0.7, rc={"lines.linewidth": 3})

# MIMIC MLP
if True:
    m_mlp_d = [1, 0.98833, 0.71295, 0.53405, 0.5045, 0.4999, 0.500365, 0.4984]
    m_mlp_p = [1, 0.98875, 0.719, 0.53165, 0.5056, 0.5034, 0.50125, 0.5015795]
    m_mlp_m = [1, 0.96365, 0.9423, 0.91745, 0.90705, 0.89865, 0.89345, 0.89115]
    m_mlp_d_std = [
        0,
        0.002053897,
        0.018943529,
        0.009477199,
        0.004680011,
        0.000684957,
        0.002380782,
        0.000864099,
    ]
    m_mlp_p_std = [
        0,
        0.001629609,
        0.017280818,
        0.011083199,
        0.00081035,
        0.001340398,
        0.001143369,
        0.000906138,
    ]
    m_mlp_m_std = [
        0,
        0.004179987,
        0.005548198,
        0.00638571,
        0.005976255,
        0.004179987,
        0.002304479,
        0.001430253,
    ]

    m_mlp_d_kg_med = [0.99935, 0.9475, 0.8825, 0.8722, 0.8736, 0.8733, 0.87275, 0.8723]
    m_mlp_p_kg_med = [0.997, 0.966, 0.934, 0.92565, 0.92485, 0.9248, 0.92405, 0.92305]
    m_mlp_m_kg_med = [1, 0.963, 0.9275, 0.92045, 0.9083, 0.8998, 0.89368, 0.8909]
    m_mlp_d_kg_med_std = [
        7.07107e-05,
        0.000707107,
        0.00212132,
        0.002262742,
        0.000989949,
        0.000848528,
        0.000353553,
        0.000424264,
    ]
    m_mlp_p_kg_med_std = [
        0,
        0.005656854,
        0.007071068,
        0.000353553,
        0.000919239,
        0.000565685,
        7.07107e-05,
        7.07107e-05,
    ]
    m_mlp_m_kg_med_std = [
        0,
        0.001414214,
        0.010606602,
        0.000212132,
        0.000424264,
        0.000424264,
        0.000113137,
        0.000141421,
    ]

    m_mlp_d_tag = [
        0.998,
        0.964,
        0.86,
        0.757,
        0.671,
        0.592,
        0.5352,
        0.5133,
    ]
    m_mlp_p_tag = [0.998, 0.97, 0.872, 0.773, 0.676, 0.611, 0.5426, 0.5145]
    m_mlp_m_tag = [
        1,
        0.973,
        0.941,
        0.913,
        0.895,
        0.89,
        0.889,
        0.8898,
    ]
    m_mlp_d_tag_std = [
        3.53553e-05,
        0.001380502,
        0.010532425,
        0.00586997,
        0.00283498,
        0.000766743,
        0.001367168,
        0.000644181,
    ]
    m_mlp_p_tag_std = [
        0,
        0.003643232,
        0.012175943,
        0.005718376,
        0.000864794,
        0.000953042,
        0.00060704,
        0.000488424,
    ]
    m_mlp_m_tag_std = [
        0,
        0.0027971,
        0.0080774,
        0.003298921,
        0.00320026,
        0.002302126,
        0.001208808,
        0.000785837,
    ]

    m_mlp_d_kg_5 = [0.999, 0.946, 0.882, 0.871, 0.872, 0.872, 0.872, 0.872]
    m_mlp_p_kg_5 = [0.997, 0.965, 0.932, 0.928, 0.926, 0.923, 0.925, 0.925]
    m_mlp_m_kg_5 = [1.0, 0.958, 0.941, 0.919, 0.908, 0.899, 0.895, 0.892]

    m_mlp_d_kg_10 = [0.999, 0.949, 0.889, 0.880, 0.882, 0.882, 0.882, 0.882]
    m_mlp_p_kg_10 = [0.997, 0.968, 0.940, 0.937, 0.936, 0.935, 0.935, 0.936]
    m_mlp_m_kg_10 = [1.0, 0.959, 0.941, 0.919, 0.908, 0.899, 0.895, 0.892]

    m_mlp_d_kg_20 = [0.999, 0.951, 0.890, 0.888, 0.887, 0.887, 0.887, 0.887]
    m_mlp_p_kg_20 = [0.997, 0.970, 0.941, 0.939, 0.939, 0.938, 0.938, 0.939]
    m_mlp_m_kg_20 = [1.0, 0.960, 0.939, 0.919, 0.907, 0.899, 0.893, 0.891]

    m_mlp_d_kg_50 = [0.999, 0.952, 0.892, 0.891, 0.890, 0.890, 0.890, 0.890]
    m_mlp_p_kg_50 = [0.997, 0.971, 0.944, 0.943, 0.944, 0.943, 0.943, 0.944]
    m_mlp_m_kg_50 = [1.0, 0.959, 0.939, 0.919, 0.907, 0.899, 0.893, 0.891]

    m_mlp_d_kg_100 = [0.935, 0.892, 0.884, 0.889, 0.881, 0.881, 0.882, 0.880]
    m_mlp_p_kg_100 = [0.969, 0.953, 0.951, 0.946, 0.943, 0.941, 0.941, 0.936]
    m_mlp_m_kg_100 = [1.0, 0.976, 0.958, 0.9380, 0.922, 0.907, 0.898, 0.894]

# MIMIC Transformer
if True:
    m_transformer_d = [
        0.938866667,
        0.917566667,
        0.889433333,
        0.87415,
        0.6793,
        0.5649,
        0.5425,
        0.5211,
    ]
    m_transformer_p = [
        0.9448,
        0.918066667,
        0.895433333,
        0.8819,
        0.708,
        0.5813,
        0.5549,
        0.5231,
    ]
    m_transformer_m = [
        0.868366667,
        0.848433333,
        0.843933333,
        0.852155,
        0.849,
        0.8157,
        0.8054,
        0.7956,
    ]

    m_transformer_d_std = [
        0.012651219,
        0.006767816,
        0.019192794,
        0.029486353,
        0.0039785,
        0.00753845,
        0.0089,
        0.00349,
    ]
    m_transformer_p_std = [
        0.013458083,
        0.005804596,
        0.010161857,
        0.03125412,
        0.009382,
        0.010237,
        0.002304479,
        0.001430253,
    ]
    m_transformer_m_std = [
        0.019209459,
        0.013596446,
        0.01636592,
        0.012947125,
        0.01328,
        0.00489,
        0.008254,
        0.009559,
    ]

    m_transformer_d_kg_med = [
        0.89494,
        0.931925,
        0.91428,
        0.863275,
        0.83618,
        0.845966667,
        0.83452,
        0.8122,
    ]
    m_transformer_p_kg_med = [
        0.90102,
        0.933075,
        0.9226475,
        0.89505,
        0.87922,
        0.898606667,
        0.8615,
        0.8415,
    ]
    m_transformer_m_kg_med = [
        0.84944,
        0.86216,
        0.862433333,
        0.83584,
        0.8525,
        0.847943333,
        0.8354,
        0.8311,
    ]

    m_transformer_d_kg_med_std = [
        0.017584596,
        0.002544766,
        0.018242302,
        0.023930925,
        0.016529543,
        0.01085833,
        0.0086,
        0.00956,
    ]
    m_transformer_p_kg_med_std = [
        0.014591504,
        0.007129458,
        0.015782069,
        0.022583253,
        0.02627769,
        0.014053332,
        0.012,
        0.00672,
    ]
    m_transformer_m_kg_med_std = [
        0.026961324,
        0.024952615,
        0.011955891,
        0.004777698,
        0.0090872537,
        0.002912668,
        0.012,
        0.012747,
    ]

    m_transformer_d_kg_10 = [
        0.8863,
        0.9286,
        0.9081,
        0.8344,
        0.8349,
        0.8452,
        0.8215,
        0.8022,
    ]
    m_transformer_p_kg_10 = [
        0.9054,
        0.9351,
        0.9167,
        0.8632,
        0.8819,
        0.8979,
        0.8524,
        0.8465,
    ]
    m_transformer_m_kg_10 = [
        0.8431,
        0.8518,
        0.862,
        0.8337,
        0.8372,
        0.8501,
        0.8322,
        0.8222,
    ]

    m_transformer_d_kg_20 = [
        0.915,
        0.9321,
        0.93072,
        0.8746,
        0.84746,
        0.8369,
        0.8354,
        0.8154,
    ]
    m_transformer_p_kg_20 = [
        0.924,
        0.9274,
        0.93909,
        0.9137,
        0.8835,
        0.88492,
        0.8625,
        0.8524,
    ]
    m_transformer_m_kg_20 = [
        0.8905,
        0.8844,
        0.8746,
        0.83346,
        0.8509,
        0.84463,
        0.8254,
        0.8212,
    ]

    m_transformer_d_kg_50 = [
        0.9106,
        0.9348,
        0.8913,
        0.8547,
        0.8456,
        0.858,
        0.8421,
        0.8254,
    ]
    m_transformer_p_kg_50 = [
        0.8965,
        0.9423,
        0.9035,
        0.8953,
        0.8918,
        0.913,
        0.845202,
        0.8325,
    ]
    m_transformer_m_kg_50 = [
        0.8606,
        0.8651,
        0.8507,
        0.8332,
        0.8444,
        0.8491,
        0.8335,
        0.8365,
    ]

    m_transformer_d_kg_100 = [
        0.8728,
        0.9322,
        0.927,
        0.8894,
        0.845,
        0.843,
        0.8251,
        0.8025,
    ]
    m_transformer_p_kg_100 = [
        0.8862,
        0.9275,
        0.9313,
        0.908,
        0.904,
        0.9035,
        0.8625,
        0.8454,
    ]
    m_transformer_m_kg_100 = [
        0.828,
        0.825,
        0.8556,
        0.843,
        0.8606,
        0.8425,
        0.8436,
        0.8321,
    ]

    m_transformer_d_tag = [0.953, 0.9492, 0.9173, 0.9115, 0.8607, 0.7828, 0.7262, 0.65]
    m_transformer_p_tag = [
        0.974,
        0.96915,
        0.9326,
        0.9218,
        0.8612,
        0.7836,
        0.6691,
        0.6102,
    ]
    m_transformer_m_tag = [
        0.887,
        0.8108,
        0.7971,
        0.7141,
        0.66464,
        0.6552,
        0.6521,
        0.6425,
    ]

    m_transformer_d_tag_std = [
        0.00369,
        0.00411,
        0.00742,
        0.00538,
        0.00804,
        0.0047,
        0.01298,
        0.003987,
    ]
    m_transformer_p_tag_std = [
        0.01091,
        0.00627,
        0.00798,
        0.00884,
        0.00486,
        0.004965,
        0.003549,
        0.004942,
    ]
    m_transformer_m_tag_std = [
        0.00979,
        0.008118,
        0.012827,
        0.01104,
        0.01045,
        0.0128,
        0.01021,
        0.006432,
    ]


# eICU MLP
if True:
    e_mlp_d = [
        0.99182,
        0.929733333,
        0.800478,
        0.610234,
        0.52524,
        0.5050955,
        0.501053333,
        0.50016,
    ]
    e_mlp_c = [0.355, 5.177, 3.245, 1.525, 1.407, 1.383, 1.377, 1.354]
    e_mlp_l = [
        1,
        0.941833333,
        0.939574,
        0.936682,
        0.933194,
        0.898575,
        0.872683333,
        0.883136,
    ]

    e_mlp_d_std = [
        0.004435313,
        0.010700156,
        0.012895132,
        0.003636177,
        0.001299231,
        0.000538963,
        0.000303535,
        0.000151658,
    ]
    e_mlp_c_std = [0.355, 5.177, 3.245, 1.525, 1.407, 1.383, 1.377, 1.354]
    e_mlp_l_std = [
        0,
        0.032788463,
        0.009022742,
        0.00470315,
        0.004383159,
        0.003221154,
        0.001483521,
        0.000887401,
    ]

    e_mlp_d_tag = [
        0.99911,
        0.95817,
        0.91927,
        0.83477,
        0.72287,
        0.633131,
        0.56748,
        0.53089,
    ]
    e_mlp_l_tag = [1.0, 0.935, 0.93135, 0.9322, 0.9412, 0.92909, 0.88224, 0.88009]

    e_mlp_d_tag_std = [
        0.0081795,
        0.00625,
        0.0022395,
        0.0035575,
        0.005616,
        0.003849,
        0.0061835,
        0.005312,
    ]
    e_mlp_l_tag_std = [
        0.003265,
        0.007958,
        0.0026685,
        0.003992,
        0.00292,
        0.006385,
        0.008106,
        0.00588,
    ]

    e_mlp_d_idlg = [0.987, 0.919, 0.817, 0.726, 0.624, 0.555, 0.501, 0.5]
    e_mlp_l_idlg = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    e_mlp_d_idlg_std = [
        0.0091745,
        0.003215,
        0.0052445,
        0.006298,
        0.007126,
        0.003609,
        0.003892,
        0.0020125,
    ]
    e_mlp_l_idlg_std = [
        0.00188,
        0.00535063,
        0.0047835,
        0.005932,
        0.0061295,
        0.0028756,
        0.004801,
        0.0098235,
    ]

    e_mlp_d_kg_5 = [
        0.991828,
        0.9829508,
        0.9803,
        0.979164,
        0.9733484,
        0.966064,
        0.9571424,
        0.959652,
    ]
    e_mlp_c_kg_5 = [12.383, 24.121, 6.640, 3.606, 2.123, 1.508, 1.386, 1.411]
    e_mlp_l_kg_5 = [1, 0.9555, 0.935234, 0.929208, 0.927894, 0.897746, 0.87908, 0.8848]

    e_mlp_d_kg_5_std = [
        0.002354978,
        0.001710527,
        0.0004,
        0.000156461,
        0.000196072,
        0.000625084,
        0.000638928,
        0.000364856,
    ]
    e_mlp_c_kg_5_std = [12.383, 24.121, 6.640, 3.606, 2.123, 1.508, 1.386, 1.411]
    e_mlp_l_kg_5_std = [
        0,
        0.01771299,
        0.006863183,
        0.005581279,
        0.002613557,
        0.0022707,
        0.001534992,
        0.000644205,
    ]

    e_mlp_d_kg_10 = [0.996, 0.985, 0.979, 0.977, 0.976, 0.974, 0.970, 0.971]
    e_mlp_c_kg_10 = [10.587, 23.925, 6.575, 3.661, 2.167, 1.519, 1.385, 1.410]
    e_mlp_l_kg_10 = [1.0, 0.935, 0.945, 0.927, 0.934, 0.900, 0.878, 0.884]

    e_mlp_d_kg_20 = [0.995, 0.983, 0.977, 0.974, 0.974, 0.974, 0.972, 0.973]
    e_mlp_c_kg_20 = [14.926, 23.831, 6.780, 3.722, 2.216, 1.528, 1.389, 1.418]
    e_mlp_l_kg_20 = [1.0, 0.912, 0.943, 0.921, 0.930, 0.900, 0.879, 0.884]

    e_mlp_d_kg_50 = [0.996, 0.983, 0.976, 0.973, 0.973, 0.974, 0.972, 0.973]
    e_mlp_c_kg_50 = [14.652, 24.471, 6.938, 3.905, 2.272, 1.534, 1.391, 1.421]
    e_mlp_l_kg_50 = [1.0, 0.915, 0.950, 0.923, 0.930, 0.900, 0.879, 0.884]

    e_mlp_d_kg_100 = [0.995, 0.983, 0.976, 0.973, 0.974, 0.973, 0.972, 0.973]
    e_mlp_c_kg_100 = [13.508, 24.322, 6.992, 3.922, 2.269, 1.541, 1.39, 1.421]
    e_mlp_l_kg_100 = [1.0, 0.925, 0.953, 0.927, 0.930, 0.900, 0.879, 0.884]

# eICU Transformer 后
if True:
    e_transformer_d = [
        0.9162,
        0.94,
        0.9211,
        0.91,
        0.8989,
        0.8862,
        0.8493,
        0.799,
    ]
    e_transformer_l = [
        0.86,
        0.905,
        0.8911,
        0.8931,
        0.8949,
        0.8857,
        0.9089,
        0.871,
    ]

    e_transformer_d_std = [
        0.00893,
        0.00159,
        0.00791,
        0.008431,
        0.00973,
        0.00283,
        0.00331,
        0.00178,
    ]
    e_transformer_l_std = [
        0.00176,
        0.00422526,
        0.00795,
        0.00828,
        0.008619,
        0.0021112,
        0.00189,
        0.009947,
    ]

    e_transformer_d_kg_5 = [
        0.933835,
        0.965125,
        0.9626655,
        0.9531125,
        0.95542,
        0.94571,
        0.940025,
        0.9455,
    ]
    e_transformer_l_kg_5 = [
        0.91845,
        0.914125,
        0.897525,
        0.9012,
        0.89,
        0.900784,
        0.92,
        0.88105,
    ]

    e_transformer_d_kg_5_std = [
        0.000252124,
        0.00057373,
        0.010496468,
        0.001163329,
        0.001559188,
        0.001926298,
        0.007005855,
        0.002186556,
    ]
    e_transformer_l_kg_5_std = [
        0.001034408,
        0.003616974,
        0.01336597,
        0.00340196,
        0.004082483,
        0.015830549,
        0,
        0.00255147,
    ]

    e_transformer_d_kg_10 = [
        0.9338,
        0.9654,
        0.9595,
        0.9514,
        0.9498,
        0.9477,
        0.946,
        0.946,
    ]
    e_transformer_l_kg_10 = [
        0.9199,
        0.916,
        0.909,
        0.9022,
        0.885,
        0.884,
        0.92,
        0.878,
    ]

    e_transformer_d_kg_20 = [
        0.93364,
        0.9652,
        0.963762,
        0.9529,
        0.9484,
        0.946,
        0.945,
        0.9451,
    ]
    e_transformer_l_kg_20 = [
        0.9184,
        0.9159,
        0.9092,
        0.9026,
        0.89,
        0.88372,
        0.92,
        0.8837,
    ]

    e_transformer_d_kg_50 = [
        0.9342,
        0.9643,
        0.9644,
        0.9523,
        0.9631,
        0.94514,
        0.9443,
        0.9468,
    ]
    e_transformer_l_kg_50 = [
        0.918,
        0.9087,
        0.8859,
        0.9038,
        0.895,
        0.9094,
        0.92,
        0.8825,
    ]

    e_transformer_d_kg_100 = [
        0.9337,
        0.9656,
        0.963,
        0.95585,
        0.9532,
        0.944,
        0.9448,
        0.9441,
    ]
    e_transformer_l_kg_100 = [
        0.9175,
        0.9159,
        0.886,
        0.8962,
        0.89,
        0.9077,
        0.92,
        0.88,
    ]

    e_transformer_d_tag = [
        0.9349,
        0.93393,
        0.9223,
        0.91716,
        0.9129,
        0.91028,
        0.90892,
        0.853,
    ]
    e_transformer_l_tag = [
        0.9018,
        0.8968,
        0.8904,
        0.88268,
        0.8825,
        0.88,
        0.869,
        0.777,
    ]

    e_transformer_d_tag_std = [
        0.009419,
        0.00484,
        0.002579,
        0.004165,
        0.004522,
        0.004388,
        0.004474,
        0.002245,
    ]
    e_transformer_l_tag_std = [
        0.002,
        0.006476,
        0.001617,
        0.003584,
        0.00364,
        0.00364,
        0.007712,
        0.0097,
    ]

    e_transformer_d_tag_std = [
        0.00694,
        0.00766,
        0.0019,
        0.00295,
        0.00671,
        0.00331,
        0.007893,
        0.008379,
    ]
    e_transformer_l_tag_std = [
        0.00453,
        0.00944,
        0.00372,
        0.0044,
        0.0022,
        0.00913,
        0.0085,
        0.00206,
    ]

    e_transformer_d_idlg = [
        0.932559,
        0.9525,
        0.925579,
        0.917115,
        0.910132,
        0.893898,
        0.861667,
        0.809624,
    ]
    e_transformer_l_idlg = [1, 1, 1, 1, 1, 1, 1, 1]

    e_transformer_d_idlg_std = [
        0.00694,
        0.00766,
        0.0019,
        0.00295,
        0.00671,
        0.00331,
        0.007893,
        0.008379,
    ]
    e_transformer_l_idlg_std = [0, 0, 0, 0, 0, 0, 0, 0]


# 0.604,0.644,0.780   0.769 0.885
base_d = [0.604] * 8
base_p = [0.644] * 8
base_m = [0.780] * 8
base_ed = [0.769] * 8
base_el = [0.885] * 8
# x_data = [1,2,4,8,16,32,64,128]
# x_label = [1,8,16,32,64,128]
x_data = [1, 2, 4, 8, 16, 32, 48, 64]
x_label = [1, 4, 8, 16, 32, 48, 64]

"""
fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(16,5))
std=[0.1]*8

high1=[m_mlp_d[i]+m_mlp_d_std[i] for i in range(8)]
low1=[m_mlp_d[i]-m_mlp_d_std[i] for i in range(8)]
high2=[m_mlp_d_kg_med[i]+m_mlp_d_kg_med_std[i] for i in range(8)]
low2=[m_mlp_d_kg_med[i]-m_mlp_d_kg_med_std[i] for i in range(8)]
high3=[m_mlp_d_tag[i]+m_mlp_d_tag_std[i] for i in range(8)]
low3=[m_mlp_d_tag[i]-m_mlp_d_tag_std[i] for i in range(8)]
sns.lineplot(x=x_data,y=m_mlp_d,label='DLG',marker='.',markersize=14,ax=ax1)
sns.lineplot(x=x_data,y=m_mlp_d_kg_med,label='GraphDLG',marker='*',markersize=14,ax=ax1)
sns.lineplot(x=x_data,y=m_mlp_d_tag,label='TAG',marker='X',markersize=10,ax=ax1)
sns.lineplot(x=x_data,y=base_d,linestyle='--',label='TopK',ax=ax1)
ax1.fill_between(x_data, high1, low1, alpha=0.2)
ax1.fill_between(x_data, high2, low2, alpha=0.2)
ax1.fill_between(x_data, high3, low3, alpha=0.2)
ax1.set_ylim(0.4,1)
ax1.set_xticks(x_label)

ax1.set_title("MIMIC-MLP-Diagnose")


high1=[m_mlp_p[i]+m_mlp_p_std[i] for i in range(8)]
low1=[m_mlp_p[i]-m_mlp_p_std[i] for i in range(8)]
high2=[m_mlp_p_kg_med[i]+m_mlp_p_kg_med_std[i] for i in range(8)]
low2=[m_mlp_p_kg_med[i]-m_mlp_p_kg_med_std[i] for i in range(8)]
high3=[m_mlp_p_tag[i]+m_mlp_p_tag_std[i] for i in range(8)]
low3=[m_mlp_p_tag[i]-m_mlp_p_tag_std[i] for i in range(8)]
sns.lineplot(x=x_data,y=m_mlp_p,label='DLG',marker='.',markersize=14,ax=ax2)
sns.lineplot(x=x_data,y=m_mlp_p_kg_med,label='GraphDLG',marker='*',markersize=14,ax=ax2)
sns.lineplot(x=x_data,y=m_mlp_p_tag,label='TAG',marker='X',markersize=10,ax=ax2)
sns.lineplot(x=x_data,y=base_p,linestyle='--',label='TopK',ax=ax2)
ax2.fill_between(x_data, high1, low1, alpha=0.2)
ax2.fill_between(x_data, high2, low2, alpha=0.2)
ax2.fill_between(x_data, high3, low3, alpha=0.2)
ax2.get_legend().remove()
ax2.set_ylim(0.4,1)
ax2.set_xticks(x_label)
ax2.set_title("MIMIC-MLP-Procedure")

high1=[m_mlp_m[i]+m_mlp_m_std[i] for i in range(8)]
low1=[m_mlp_m[i]-m_mlp_m_std[i] for i in range(8)]
high2=[m_mlp_m_kg_med[i]+m_mlp_m_kg_med_std[i] for i in range(8)]
low2=[m_mlp_m_kg_med[i]-m_mlp_m_kg_med_std[i] for i in range(8)]
high3=[m_mlp_m_tag[i]+m_mlp_m_tag_std[i] for i in range(8)]
low3=[m_mlp_m_tag[i]-m_mlp_m_tag_std[i] for i in range(8)]
ax3.fill_between(x_data, high1, low1, alpha=0.2)
ax3.fill_between(x_data, high2, low2, alpha=0.2)
ax3.fill_between(x_data, high3, low3, alpha=0.2)
sns.lineplot(x=x_data,y=m_mlp_m,label='DLG',marker='.',markersize=14,ax=ax3)
sns.lineplot(x=x_data,y=m_mlp_m_kg_med,label='GraphDLG',marker='*',markersize=14,ax=ax3)
sns.lineplot(x=x_data,y=m_mlp_m_tag,label='TAG',marker='X',markersize=10,ax=ax3)
sns.lineplot(x=x_data,y=base_m,linestyle='--',label='TopK',ax=ax3)
ax3.get_legend().remove()
ax3.set_ylim(0.75,1)
ax3.set_xticks(x_label)
ax3.set_title("MIMIC-MLP-Medicine")
"""

"""
fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(16,5))

sns.lineplot(x=x_data,y=m_transformer_d_kg_med,label='5%',ax=ax1,marker='.',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_d_kg_10,label='10%',ax=ax1,marker='X',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_d_kg_20,label='20%',ax=ax1,marker='^',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_d_kg_50,label='50%',ax=ax1,marker='*',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_d_kg_100,label='100%',ax=ax1,marker='d',markersize=10)
ax1.set_xticks(x_label)
ax1.set_ylim(0.75,1)
ax1.set_title("MIMIC-Transformer-Diagnose-Scale")

sns.lineplot(x=x_data,y=m_transformer_p_kg_med,label='5%',ax=ax2,marker='.',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_p_kg_10,label='10%',ax=ax2,marker='X',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_p_kg_20,label='20%',ax=ax2,marker='^',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_p_kg_50,label='50%',ax=ax2,marker='*',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_p_kg_100,label='100%',ax=ax2,marker='d',markersize=10)
ax2.get_legend().remove()
ax2.set_xticks(x_label)
ax2.set_ylim(0.75,1)
ax2.set_title("MIMIC-Transformer-Procedure-Scale")

sns.lineplot(x=x_data,y=m_transformer_m_kg_med,label='5%',ax=ax3,marker='.',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_m_kg_10,label='10%',ax=ax3,marker='X',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_m_kg_20,label='20%',ax=ax3,marker='^',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_m_kg_50,label='50%',ax=ax3,marker='*',markersize=10)
sns.lineplot(x=x_data,y=m_transformer_m_kg_100,label='100%',ax=ax3,marker='d',markersize=10)
ax3.get_legend().remove()
ax3.set_xticks(x_label)
ax3.set_ylim(0.75,1)
ax3.set_title("MIMIC-Transformer-Medcine-Scale")
"""

"""
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(12,5))

high1=[e_mlp_d[i]+e_mlp_d_std[i] for i in range(8)]
low1=[e_mlp_d[i]-e_mlp_d_std[i] for i in range(8)]
high2=[e_mlp_d_kg_5[i]+e_mlp_d_kg_5_std[i] for i in range(8)]
low2=[e_mlp_d_kg_5[i]-e_mlp_d_kg_5_std[i] for i in range(8)]
high3=[e_mlp_d_tag[i]+e_mlp_d_tag_std[i] for i in range(8)]
low3=[e_mlp_d_tag[i]-e_mlp_d_tag_std[i] for i in range(8)]
high4=[e_mlp_d_idlg[i]+e_mlp_d_idlg_std[i] for i in range(8)]
low4=[e_mlp_d_idlg[i]-e_mlp_d_idlg_std[i] for i in range(8)]
sns.lineplot(x=x_data,y=e_mlp_d,label='DLG',marker='.',markersize=14,ax=ax1)
sns.lineplot(x=x_data,y=e_mlp_d_kg_5,label='GraphDLG',marker='*',markersize=14,ax=ax1)
sns.lineplot(x=x_data,y=e_mlp_d_tag,label='TAG',marker='X',markersize=10,ax=ax1)
sns.lineplot(x=x_data,y=e_mlp_d_idlg,label='iDLG',marker='^',markersize=14,ax=ax1)
sns.lineplot(x=x_data,y=base_ed,linestyle='--',label='TopK',ax=ax1)
ax1.fill_between(x_data, high1, low1, alpha=0.2)
ax1.fill_between(x_data, high2, low2, alpha=0.2)
ax1.fill_between(x_data, high3, low3, alpha=0.2)
ax1.fill_between(x_data, high4, low4, alpha=0.2)
ax1.legend(loc='upper right')
ax1.set_xticks(x_label)
ax1.set_title("eICU-MLP-Data")

high1=[e_mlp_l[i]+e_mlp_l_std[i] for i in range(8)]
low1=[e_mlp_l[i]-e_mlp_l_std[i] for i in range(8)]
high2=[e_mlp_l_kg_5[i]+e_mlp_l_kg_5_std[i] for i in range(8)]
low2=[e_mlp_l_kg_5[i]-e_mlp_l_kg_5_std[i] for i in range(8)]
high3=[e_mlp_l_tag[i]+e_mlp_l_tag_std[i] for i in range(8)]
low3=[e_mlp_l_tag[i]-e_mlp_l_tag_std[i] for i in range(8)]
high4=[e_mlp_l_idlg[i]+e_mlp_l_idlg_std[i] for i in range(8)]
low4=[e_mlp_l_idlg[i]-e_mlp_l_idlg_std[i] for i in range(8)]
sns.lineplot(x=x_data,y=e_mlp_l,label='DLG',marker='.',markersize=14,ax=ax2)
sns.lineplot(x=x_data,y=e_mlp_l_kg_5,label='GraphDLG',marker='*',markersize=14,ax=ax2)
sns.lineplot(x=x_data,y=e_mlp_l_tag,label='TAG',marker='X',markersize=10,ax=ax2)
sns.lineplot(x=x_data,y=e_mlp_l_idlg,label='iDLG',marker='^',markersize=14,ax=ax2)
sns.lineplot(x=x_data,y=base_el,linestyle='--',label='TopK',ax=ax2)
ax2.get_legend().remove()
ax2.set_ylim(0.75)
ax2.set_xticks(x_label)
ax2.set_title("eICU-MLP-Label")
"""


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.lineplot(
    x=x_data, y=e_transformer_d_kg_5, label="5%", ax=ax1, marker=".", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_d_kg_10, label="10%", ax=ax1, marker="X", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_d_kg_20, label="20%", ax=ax1, marker="^", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_d_kg_50, label="50%", ax=ax1, marker="*", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_d_kg_100, label="100%", ax=ax1, marker="d", markersize=10
)
ax1.set_ylim(0.88, 1)
ax1.set_xticks(x_label)
ax1.legend(loc="lower right")
ax1.set_title("eICU-Transformer-Data-Scale")

sns.lineplot(
    x=x_data, y=e_transformer_l_kg_5, label="5%", ax=ax2, marker=".", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_l_kg_10, label="10%", ax=ax2, marker="X", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_l_kg_20, label="20%", ax=ax2, marker="^", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_l_kg_50, label="50%", ax=ax2, marker="*", markersize=10
)
sns.lineplot(
    x=x_data, y=e_transformer_l_kg_100, label="100%", ax=ax2, marker="d", markersize=10
)
ax2.set_ylim(0.79, 1)
ax2.set_xticks(x_label)
ax2.get_legend().remove()
ax2.set_title("eICU-Transformer-Label-Scale")

plt.tight_layout()
fig.savefig("imgs/eicu_transformer_scale.png", dpi=400)


### 横轴
x_data_mlp = [1, 2, 4, 8, 16, 32, 64, 128]
x_data_transformer = [1, 2, 4, 8, 16, 32, 48, 64]


###横轴显示的刻度
x_label_mlp = [1, 8, 16, 32, 64, 128]
x_label_transformer = [1, 8, 16, 32, 48, 64]

######第一张大图
######第一张子图不变还是折线图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
high1 = [e_mlp_d[i] + e_mlp_d_std[i] for i in range(8)]
low1 = [e_mlp_d[i] - e_mlp_d_std[i] for i in range(8)]
high2 = [e_mlp_d_kg_5[i] + e_mlp_d_kg_5_std[i] for i in range(8)]
low2 = [e_mlp_d_kg_5[i] - e_mlp_d_kg_5_std[i] for i in range(8)]
high3 = [e_mlp_d_tag[i] + e_mlp_d_tag_std[i] for i in range(8)]
low3 = [e_mlp_d_tag[i] - e_mlp_d_tag_std[i] for i in range(8)]
high4 = [e_mlp_d_idlg[i] + e_mlp_d_idlg_std[i] for i in range(8)]
low4 = [e_mlp_d_idlg[i] - e_mlp_d_idlg_std[i] for i in range(8)]
sns.lineplot(x=x_data_mlp, y=e_mlp_d, label="DLG", marker=".", markersize=14, ax=ax1)
sns.lineplot(
    x=x_data_mlp, y=e_mlp_d_kg_5, label="GraphDLG", marker="*", markersize=14, ax=ax1
)
sns.lineplot(
    x=x_data_mlp, y=e_mlp_d_tag, label="TAG", marker="X", markersize=10, ax=ax1
)
sns.lineplot(
    x=x_data_mlp, y=e_mlp_d_idlg, label="iDLG", marker="^", markersize=14, ax=ax1
)
sns.lineplot(x=x_data_mlp, y=base_ed, linestyle="--", label="TopK", ax=ax1)
ax1.fill_between(x_data_mlp, high1, low1, alpha=0.2)
ax1.fill_between(x_data_mlp, high2, low2, alpha=0.2)
ax1.fill_between(x_data_mlp, high3, low3, alpha=0.2)
ax1.fill_between(x_data_mlp, high4, low4, alpha=0.2)
ax1.legend(loc="upper right")
ax1.set_xticks(x_label_mlp)
ax1.set_title("eICU-MLP-Data")

######第二张子图变成柱状图
high1 = [e_mlp_l[i] + e_mlp_l_std[i] for i in range(8)]
low1 = [e_mlp_l[i] - e_mlp_l_std[i] for i in range(8)]
high2 = [e_mlp_l_kg_5[i] + e_mlp_l_kg_5_std[i] for i in range(8)]
low2 = [e_mlp_l_kg_5[i] - e_mlp_l_kg_5_std[i] for i in range(8)]
high3 = [e_mlp_l_tag[i] + e_mlp_l_tag_std[i] for i in range(8)]
low3 = [e_mlp_l_tag[i] - e_mlp_l_tag_std[i] for i in range(8)]
high4 = [e_mlp_l_idlg[i] + e_mlp_l_idlg_std[i] for i in range(8)]
low4 = [e_mlp_l_idlg[i] - e_mlp_l_idlg_std[i] for i in range(8)]
sns.barplot(x=x_data_mlp, y=e_mlp_l, label="DLG", ax=ax2)
sns.barplot(x=x_data_mlp, y=e_mlp_l_kg_5, label="GraphDLG", ax=ax2)
sns.barplot(x=x_data_mlp, y=e_mlp_l_tag, label="TAG", ax=ax2)
sns.barplot(x=x_data_mlp, y=e_mlp_l_idlg, label="iDLG", ax=ax2)
sns.lineplot(x=x_data_mlp, y=base_el, linestyle="--", label="TopK", ax=ax2)
ax2.get_legend().remove()
ax2.set_ylim(0.75)
ax2.set_xticks(x_label_mlp)
ax2.set_title("eICU-MLP-Label")

plt.tight_layout()
fig.savefig("test1.png", dpi=400)


######第二张大图
######第一张子图不变还是折线图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
high1 = [e_transformer_d[i] + e_transformer_d_std[i] for i in range(8)]
low1 = [e_transformer_d[i] - e_transformer_d_std[i] for i in range(8)]
high2 = [e_transformer_d_kg_5[i] + e_transformer_d_kg_5_std[i] for i in range(8)]
low2 = [e_transformer_d_kg_5[i] - e_transformer_d_kg_5_std[i] for i in range(8)]
high3 = [e_transformer_d_tag[i] + e_transformer_d_tag_std[i] for i in range(8)]
low3 = [e_transformer_d_tag[i] - e_transformer_d_tag_std[i] for i in range(8)]
high4 = [e_transformer_d_idlg[i] + e_transformer_d_idlg_std[i] for i in range(8)]
low4 = [e_transformer_d_idlg[i] - e_transformer_d_idlg_std[i] for i in range(8)]
sns.lineplot(
    x=x_data_transformer,
    y=e_transformer_d,
    label="DLG",
    marker=".",
    markersize=14,
    ax=ax1,
)
sns.lineplot(
    x=x_data_transformer,
    y=e_transformer_d_kg_5,
    label="GraphDLG",
    marker="*",
    markersize=14,
    ax=ax1,
)
sns.lineplot(
    x=x_data_transformer,
    y=e_transformer_d_tag,
    label="TAG",
    marker="X",
    markersize=10,
    ax=ax1,
)
sns.lineplot(
    x=x_data_transformer,
    y=e_transformer_d_idlg,
    label="iDLG",
    marker="^",
    markersize=14,
    ax=ax1,
)
sns.lineplot(x=x_data_transformer, y=base_ed, linestyle="--", label="TopK", ax=ax1)
ax1.fill_between(x_data_transformer, high1, low1, alpha=0.2)
ax1.fill_between(x_data_transformer, high2, low2, alpha=0.2)
ax1.fill_between(x_data_transformer, high3, low3, alpha=0.2)
ax1.fill_between(x_data_transformer, high4, low4, alpha=0.2)
ax1.legend(loc="upper right")
ax1.set_xticks(x_label_transformer)
ax1.set_title("eICU-Transformer-Data")

######第二张子图变成条形图
high1 = [e_transformer_l[i] + e_transformer_l_std[i] for i in range(8)]
low1 = [e_transformer_l[i] - e_transformer_l_std[i] for i in range(8)]
high2 = [e_transformer_l_kg_5[i] + e_transformer_l_kg_5_std[i] for i in range(8)]
low2 = [e_transformer_l_kg_5[i] - e_transformer_l_kg_5_std[i] for i in range(8)]
high3 = [e_transformer_l_tag[i] + e_transformer_l_tag_std[i] for i in range(8)]
low3 = [e_transformer_l_tag[i] - e_transformer_l_tag_std[i] for i in range(8)]
high4 = [e_transformer_l_idlg[i] + e_transformer_l_idlg_std[i] for i in range(8)]
low4 = [e_transformer_l_idlg[i] - e_transformer_l_idlg_std[i] for i in range(8)]
sns.barplot(x=x_data_transformer, y=e_transformer_l, label="DLG", ax=ax2)
sns.barplot(x=x_data_transformer, y=e_transformer_l_kg_5, label="GraphDLG", ax=ax2)
sns.barplot(x=x_data_transformer, y=e_transformer_l_tag, label="TAG", ax=ax2)
sns.barplot(x=x_data_transformer, y=e_transformer_l_idlg, label="iDLG", ax=ax2)
sns.barplot(x=x_data_transformer, y=base_el, linestyle="--", label="TopK", ax=ax2)
ax2.get_legend().remove()
ax2.set_ylim(0.75)
ax2.set_xticks(x_label_transformer)
ax2.set_title("eICU-Transformer-Label")

plt.tight_layout()
fig.savefig("test2.png", dpi=400)
