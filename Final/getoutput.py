#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, fbeta_score, f1_score
import sys

import time, datetime
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str, required=True)
parser.add_argument("-outfile", type=str, required=True)
args = parser.parse_args()


# mapping dict to map the categories to numerical values #
mapping_dict = {
'ind_empleado' 	: {'N':0, -99:1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo' : {'V':0, 'H':1, -99:2},
'ind_nuevo' 	: {0.0:0, 1.0:1, -99.0:2},
'indrel'		: {1.0:0, 99.0:1, -99.0:2},
'indrel_1mes'	: {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},
'tiprel_1mes'	: {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'indresi': {-99:0, 'S':1, 'N':2},
'indext': {-99:0, 'S':1, 'N':2},
#'conyuemp': {-99:0, 'S':1, 'N':2},
'indfall': {-99:0, 'S':1, 'N':2},
#'tipodom': {-99.0:0, 1.0:1},
'ind_actividad_cliente' : {0.0:0, 1.0:1, -99.0:2},
'segmento': {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:3},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11},
'nomprov' : {'ZARAGOZA': 2, 'BURGOS': 11, 'GRANADA': 46, 'MADRID': 18, 'CIUDAD REAL': 1, 'GIRONA': 7, 'TARRAGONA': 50, 'LEON': 4, 'SORIA': 20, 'SANTA CRUZ DE TENERIFE': 48, 'CEUTA': 52, 'HUESCA': 12, 'VALLADOLID': 24, 'LERIDA': 17, 'ZAMORA': 8, 'CUENCA': 31, 'RIOJA, LA': 34, 'TERUEL': 27, 'PONTEVEDRA': 25, 'MELILLA': 49, 'CORDOBA': 44, 'SEVILLA': 21, -99: 39, 'ALICANTE': 19, 'CASTELLON': 33, 'OURENSE': 29, 'VALENCIA': 26, 'CORU\xc3\x91A, A':28, 'CORUNA, A': 28, 'HUELVA': 45, 'ALBACETE': 35, 'JAEN': 30, 'CADIZ': 38, 'BADAJOZ': 36, 'TOLEDO': 3, 'AVILA': 14, 'BARCELONA': 9, 'SEGOVIA': 15, 'NAVARRA': 13, 'MALAGA': 0, 'SALAMANCA': 10, 'PALENCIA': 42, 'ALMERIA': 40, 'MURCIA': 37, 'GUADALAJARA': 41, 'ASTURIAS': 47, 'BALEARS, ILLES': 23, 'ALAVA': 51, 'LUGO': 16, 'CANTABRIA': 22, 'CACERES': 6, 'PALMAS, LAS': 43, 'GIPUZKOA': 5, 'BIZKAIA': 32}
}

# renta dict
renta_dict = {'ALBACETE': 76895,  'ALICANTE': 60562,  'ALMERIA': 77815,  'ASTURIAS': 83995,  'AVILA': 78525,  'BADAJOZ': 60155,  'BALEARS, ILLES': 114223,  'BARCELONA': 135149,  'BURGOS': 87410, 'NAVARRA' : 101850, 'CACERES': 78691,  'CADIZ': 75397,  'CANTABRIA': 87142,  'CASTELLON': 70359,  'CEUTA': 333283, 'CIUDAD REAL': 61962,  'CORDOBA': 63260,  'CORU\xc3\x91A, A': 103567,  'CUENCA': 70751,  'GIRONA': 100208,  'GRANADA': 80489, 'GUADALAJARA': 100635,  'HUELVA': 75534,  'HUESCA': 80324,  'JAEN': 67016,  'LEON': 76339,  'LERIDA': 59191,  'LUGO': 68219,  'MADRID': 141381,  'MALAGA': 89534,  'MELILLA': 116469, 'GIPUZKOA': 101850, 'MURCIA': 68713,  'OURENSE': 78776,  'PALENCIA': 90843,  'PALMAS, LAS': 78168,  'PONTEVEDRA': 94328,  'RIOJA, LA': 91545,  'SALAMANCA': 88738,  'SANTA CRUZ DE TENERIFE': 83383, 'ALAVA': 101850, 'BIZKAIA' : 101850, 'SEGOVIA': 81287,  'SEVILLA': 94814,  'SORIA': 71615,  'TARRAGONA': 81330,  'TERUEL': 64053,  'TOLEDO': 65242,  'UNKNOWN': 103689,  'VALENCIA': 73463,  'VALLADOLID': 92032,  'ZAMORA': 73727,  'ZARAGOZA': 98827}

# dtype list for columns to be used for reading #
dtype_list = {'ind_cco_fin_ult1': 'float16', 'ind_deme_fin_ult1': 'float16', 'ind_aval_fin_ult1': 'float16', 'ind_valo_fin_ult1': 'float16', 'ind_reca_fin_ult1': 'float16', 'ind_ctju_fin_ult1': 'float16', 'ind_cder_fin_ult1': 'float16', 'ind_plan_fin_ult1': 'float16', 'ind_fond_fin_ult1': 'float16', 'ind_hip_fin_ult1': 'float16', 'ind_pres_fin_ult1': 'float16', 'ind_nomina_ult1': 'float16', 'ind_cno_fin_ult1': 'float16', 'ncodpers': 'int64', 'ind_ctpp_fin_ult1': 'float16', 'ind_ahor_fin_ult1': 'float16', 'ind_dela_fin_ult1': 'float16', 'ind_ecue_fin_ult1': 'float16', 'ind_nom_pens_ult1': 'float16', 'ind_recibo_ult1': 'float16', 'ind_deco_fin_ult1': 'float16', 'ind_tjcr_fin_ult1': 'float16', 'ind_ctop_fin_ult1': 'float16', 'ind_viv_fin_ult1': 'float16', 'ind_ctma_fin_ult1': 'float16', 'conyuemp': str, 'indrel_1mes': str}

# target columns to predict #
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
# 1219
target_cols = target_cols[2:]
print 'length of target_cols', len(target_cols)

# numerical columns to use #
numerical_cols = ['age', 'antiguedad', 'renta']

# categorical columns to use #
cols_to_use = mapping_dict.keys()

# hsieh add
hsieh_cols = ['ncodpers']
#time_cols = ['fecha_dato','fecha_alta','ult_fec_cli_1t']
#time_cols = ['fecha_alta', 'ult_fec_cli_1t']
time_cols = ['fecha_dato','fecha_alta']

total_use_cols = cols_to_use+numerical_cols+hsieh_cols+time_cols
#total_use_cols = cols_to_use+numerical_cols

# transfer time to POSIX
def timestrToStamp(element):
    if isinstance(element, basestring):
        return time.mktime(datetime.datetime.strptime(element,'%Y-%m-%d').timetuple())
    else:
        return "FUCK"

def getMonth(element):
    return int(element.split('-')[1])

def myloadData(filepath, dtype_list, mapping_dict, needLabel=True):
    print "Loading and processing ", filepath
    # read csv include first line
    data = pd.read_csv(filepath, dtype = dtype_list)
    # remove those whose fecha_alta is NaN!!!
    data = data[ data["fecha_alta"].isnull()==False ]
    # fix continuous column
    data.antiguedad = pd.to_numeric(data.antiguedad,errors="coerce")
    data.age = pd.to_numeric(data.age,errors="coerce")
    # transform renta
    for rent_keys in renta_dict.keys():
        data.loc[(data.nomprov == rent_keys) & (data.renta.isnull()),["renta"]] = renta_dict[rent_keys]
    data.renta = pd.to_numeric(data.renta,errors="coerce")
    data.renta = data.renta.fillna(renta_dict['UNKNOWN'])
    data.age = data.age.fillna(40) # mean_age
    data.antiguedad = data.antiguedad.fillna(0) # mean_antiguedad
    data.loc[data.antiguedad<0, "antiguedad"] = 0
    data[numerical_cols] = data[numerical_cols].apply(lambda x: (x - x.mean())/( x.std(ddof=0) ) )
    # fix missing value
    #data.loc[data["ult_fec_cli_1t"].isnull()==True, "ult_fec_cli_1t"] = data["ult_fec_cli_1t"].mode()[0]
    
    # tran time columns
    #data["fecha_alta"] = data["fecha_alta"].apply(timestrToStamp)
    #data["fecha_dato"] = data["fecha_dato"].apply(timestrToStamp)
    data["fecha_alta"] = data["fecha_alta"].apply(getMonth)
    data["fecha_dato"] = data["fecha_dato"].apply(getMonth)
    #data[time_cols] = data[time_cols].apply(lambda x: ( x - x.mean() )/( x.std(ddof=0) ) )
    
    # categorical value
    for col_ind, col in enumerate(cols_to_use):
        data[col] = data[col].fillna(-99)
        data[col] = data[col].apply(lambda x: mapping_dict[col][x])
    data_train = data[total_use_cols].values
    if needLabel == True:
        label_df = data[target_cols].fillna(0)
        label_train = label_df.values
        label_train = label_train.astype('int')

    data_train = data_train.astype('float32')

    #print data[total_use_cols].isnull().any()
    print "Still null cols ", sum(data[total_use_cols].isnull().any())
    #print label_df.isnull().any()
    #print len(np.unique(data['ncodpers'].values))
    del data
    print "Processing done"
    if needLabel == False:
        return data_train
    return data_train, label_train

# take pre month and now month to compute what did they newly buy this month
# each column has only one newly buy products!!
def lessIsMore(data_old, label_old, data_new):
    print "Less is More running..."
    # cols_to_use has 14, ncodpers is the 18th feature
    ncodpers_pos = len(cols_to_use)+len(numerical_cols)
    print 'Ncodpers is at ', ncodpers_pos
    cus_id = data_new[:, ncodpers_pos]
    old_cus_id = set(data_old[:, ncodpers_pos])
    old_cus_dict = dict( (x, k) for (k, x) in enumerate(data_old[:, ncodpers_pos]) )
    new_buy_data =  []
    for i,cus in enumerate(cus_id):
       if cus in old_cus_id:
            prev_products = label_old[old_cus_dict[cus], :]
            new_buy_data.append(np.append(data_new[i,:], prev_products))
    new_buy_data = np.array(new_buy_data)
    print "Less is More finished..."
    return new_buy_data

#
def loadPrevLabels(filepath, dtype_list):
    print "Loading prev label from ", filepath
    data = pd.read_csv(filepath, dtype = dtype_list)
    data = data[ data["fecha_alta"].isnull()==False ]
    # only get ncodpers
    data_train = data['ncodpers'].values.astype('int')
    label_train = data[target_cols].fillna(0).values.astype('int')

    print 'Ncodpers null: ', data["ncodpers"].isnull().any()
    print "Total buy cus : ", len(np.unique(data_train[:]))
    del data
    return data_train, label_train

# add prev month products
def addPrevProducts(data_new, oldpers, label_old):
    print "Add prev products..."
    ncodpers_pos = len(cols_to_use)+len(numerical_cols)
    cus_id = data_new[:, ncodpers_pos]
    # old at first col
    old_cus_id = set(oldpers[:])
    old_cus_dict = dict( (x, k) for (k, x) in enumerate(oldpers[:]) )
    new_data = []
    for i, cus in enumerate(cus_id):
        if cus in old_cus_id:
            prev_products = label_old[old_cus_dict[cus], :]
            new_data.append( np.append(data_new[i,:], prev_products) )
        else:
            new_data.append( np.append(data_new[i,:], [0]*len(target_cols) ) )
    new_data = np.array(new_data)
    print "Add finished"
    print "Total buy cus : ", len(np.unique(new_data[:,ncodpers_pos]))
    return new_data

data_17, label_17 = myloadData("./data/month17.csv", dtype_list, mapping_dict)
data_test = myloadData("./data/test_ver2.csv", dtype_list, mapping_dict, False)
new_data = lessIsMore(data_17, label_17, data_test)
del data_17, label_17, data_test

data_16, label_16 = loadPrevLabels("./data/month16.csv", dtype_list)
new_data = addPrevProducts(new_data, data_16, label_16)
del data_16, label_16

data_15, label_15 = loadPrevLabels("./data/month15.csv", dtype_list)
new_data = addPrevProducts(new_data, data_15, label_15)
del data_15, label_15

data_14, label_14 = loadPrevLabels("./data/month14.csv", dtype_list)
new_data = addPrevProducts(new_data, data_14, label_14)
del data_14, label_14

data_13, label_13 = loadPrevLabels("./data/month13.csv", dtype_list)
new_data = addPrevProducts(new_data, data_13, label_13)
del data_13, label_13

d_test = xgb.DMatrix(new_data)

bst = xgb.Booster()
bst.load_model(args.model)
#bst.load_model("./model/XGBmodel_lessIsMore")
preds = bst.predict(d_test)

print("Getting the top products..")
target_cols = np.array(target_cols)
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)[:,:7]
test_id = np.array(pd.read_csv("./data/test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
final_preds = [" ".join(list(target_cols[pred])) for pred in preds]
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv(args.outfile, index=False)
#out_df.to_csv("ppap", index=False)

