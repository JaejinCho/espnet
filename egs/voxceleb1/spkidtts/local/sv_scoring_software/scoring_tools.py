# -*- coding: utf-8 -*-

"""
This script gathers a set of functions useful for generating database and filtering.

"""

__author__  = "Timothee Kheyrkhah"
__email__   = "timothee.kheyrkhah@nist.gov"
__version__ = "4.2"

import numpy as np
import numpy.lib.recfunctions as rf


###############################################################################
# Database generation functions

def name_dtypes(file):
    """ read the names of the columns at the first of a file
        given its path
    """
    with open(file,'r') as f:
        columns = f.readline().split()
    return tuple(columns)

def get_dtype(table,name):
    """ get the dtype of a field in a table (recarray)
        given its name
    """
    return table.dtype.fields[name][0].descr[0][1]

def get_dtypes(table,names):
    """ same as get_dtype but for a list of names
    """
    dTypes = []
    for name in names:
        dTypes += [table.dtype.fields[name][0].descr[0][1]]
    return dTypes


def rSelect(table,field,value,string_value=False):
    """ Make a selection in table (recarray) base on the given value
        of one field.
        If value is a string, use string_value = True
    """
    if string_value == True:
        return table[table[field].astype('U')==value]
    return table[table[field]==value]

def fusion_table(rcarr1, rcarr2, columns, p_key):
    """
    Make a append of a list of columns of rcarr2 to rcarray1
    given the primary key

    Parameters
    ----------
    rcarr1 : recarray
        recarray use as a base
    rcarr2 : recarray
        recarray that we whant to append to rcarr1
    columns : list
        string list of the name of the fields of the rcarr2 that we when to
        append to rcarr1
    p_key : string
        field used as a primary key for both recarray
    """
    dTypes = []
    stck_columns = []
    for c in columns:
        dTypes += [rcarr2.dtype.fields[c][0].descr[0][1]]
    for dt in dTypes:
        stck_columns += [np.ndarray((rcarr1.size,),dtype=dt)]
    for i,data in enumerate(rcarr1):
#        selection = rcarr2[rcarr2[p_key] == data[p_key]]
        selection = rSelect(rcarr2,p_key,data[p_key])
        for j,stck_c in enumerate(stck_columns):
            stck_c[i] = selection[columns[j]][0]
    rcarr1 = rf.append_fields(rcarr1, columns, data=stck_columns, dtypes=dTypes,\
    fill_value=-1, usemask=False, asrecarray=True)
    return rcarr1

def fusion_and_hash_segs_calls(rcarr1, rcarr2, columns, p_key):
    """
    Make a append of a list of columns of rcarr2 to rcarray1
    given the primary key

    Parameters
    ----------
    rcarr1 : recarray
        recarray use as a base
    rcarr2 : recarray
        recarray that we whant to append to rcarr1
    columns : list
        string list of the name of the fields of the rcarr2 that we when to
        append to rcarr1
    p_key : string
        field used as a primary key for both recarray
    """
    dTypes = []
    Dic_seg_phone = dict()
    stck_columns = []
    for c in columns:
        dTypes += [rcarr2.dtype.fields[c][0].descr[0][1]]
    for dt in dTypes:
        stck_columns += [np.ndarray((rcarr1.size,),dtype=dt)]
    for i,data_seg in enumerate(rcarr1):
#        selection = rcarr2[rcarr2[p_key] == data[p_key]]
        selection_call = rSelect(rcarr2,p_key,data_seg[p_key])
        seg_id = data_seg["segment"]
        phone = selection_call["phone_id"][0]
        Dic_seg_phone[seg_id] = phone
        for j,stck_c in enumerate(stck_columns):
            stck_c[i] = selection_call[columns[j]][0]
    rcarr1 = rf.append_fields(rcarr1, columns, data=stck_columns, dtypes=dTypes,\
    fill_value=-1, usemask=False, asrecarray=True)
    return rcarr1, Dic_seg_phone



def create_enrollments_dict(path_enrollments_file):
    """ Generate an ordered dictionnary containing data of the enrollements
        {'modelid', {'segments': ['seg_1',...], 'side': 'a'}}
        It gathers for each model all the enrollements segments.
    """
    from collections import OrderedDict
    Enrollments = OrderedDict()
    with open(path_enrollments_file,'r') as f:
        lines = f.read().splitlines()
        keys = lines[0].split('\t')
        if keys == ['modelid', 'segment', 'side']:
            for line in lines[1:]:
                modelid, segment, side = line.split('\t')
                if modelid not in Enrollments:
                    Enrollments[modelid] = {'segments':[segment],'side':side}
                else:
                    Enrollments[modelid]['segments'] += [segment]
        else:
            print("Problem with the columns names/order of the enrollment file.\nShould be {}".format(keys))
    return Enrollments

def create_model_table(Enrollments_dict,segments):
    """
    Generate the models recarray and the phone number hashtable

    Parameters
    ----------
    Enrollments_dict : dict
        Dictionnary created by create_enrollments_dict()
    segments : recarray
        array containing informations all the segments

    Notes
    -----
    * Here the names of the columns are fixed and need to correspond
      to others data files
    """
    Hash_Model_phone = dict()
    models_columns = ('modelid','subject_id','sex','language_id','phone_id','nb_enrollment_segments')
    models_formats = ('S16',get_dtype(segments,'subject_id'),\
                           get_dtype(segments,'sex'),\
                           get_dtype(segments,'language_id'),\
                           get_dtype(segments,'phone_id'),\
                           'i4')
    models_dtype = {'names':models_columns,'formats':models_formats}
    models = np.recarray((len(Enrollments_dict),),dtype=models_dtype)

    for i, (modelid, data) in enumerate(Enrollments_dict.items()):
        first_enrollment_segment = data['segments'][0]
        row_segment = segments[segments['segment'].astype('U') == first_enrollment_segment]

        subject_id = row_segment['subject_id'][0]
        sex = row_segment['sex'][0]
        language_id = row_segment['language_id'][0]
        phone_id = row_segment['phone_id'][0]
        nb_segs = len(data['segments'])
        
        models[i] = np.array([(modelid,subject_id,sex,language_id,phone_id,nb_segs)],dtype=models_dtype)
        Hash_Model_phone[modelid] = phone_id

    return models, Hash_Model_phone


def join_duplicate(arr1,arr2,p_key,name):
    """
    Join arrays `arr1` and `arr2` on key `p_key`
    with duplication on the primary key. (append)

    Parameters
    ----------
    arr1, arr2 : recarray
        Structured arrays.
    p_key : string
        Primary key.
    name : string
        field name of arr2 that is added to a arr1

    Notes
    -----
    * Checks if name is in arr1 and increments name in order to
      avoid collisions.
      For example : ['field_B', 'field_B'] will become ['field_B', 'field_B2']
    """
    column_dtype = get_dtype(arr2,name)
    k = 2
    name_2 = name
    while name_2 in arr1.dtype.names:
        name_2 += str(k)
        k += 1
    stack_column = np.ndarray((arr1.size,),dtype=column_dtype)
    for i,data in enumerate(arr1):
        stack_column[i] = arr2[arr2[p_key] == data[p_key]][name][0]
    join = rf.append_fields(arr1, name_2, \
                            data=stack_column, \
                            dtypes=column_dtype, \
                            fill_value=-1,usemask=False, asrecarray=True)
    return join

def hashtable_segment_phone(Segments):
    Dic = dict()
    for i in range(Segments.size):
        segment_id = Segments[i][0]
        phone_id = Segments[i][7]
        Dic[segment_id] = phone_id
    return Dic

def create_data_arrays(path_call_sides,\
                       path_calls,\
                       path_subjects,\
                       path_segments_file,\
                       path_enrollments_file,\
                       path_reference_file):

    """ Create the structured arrays gathering all the data about
        segments, models, trials, etc from the data files
    """

    ###################################
    # Reading and fusionning tables

    # Generating record arrays
    call_sides_array = np.genfromtxt(path_call_sides,dtype=None,skip_header=1)
    calls_array = np.genfromtxt(path_calls,dtype=None,skip_header=1)
    subjects_array = np.genfromtxt(path_subjects,dtype=None,skip_header=1)
    segments_array = np.genfromtxt(path_segments_file,dtype=None,skip_header=1)

    # Naming fields
    call_sides_array.dtype.names = name_dtypes(path_call_sides)
    calls_array.dtype.names = name_dtypes(path_calls)
    subjects_array.dtype.names = name_dtypes(path_subjects)
    segments_array.dtype.names = name_dtypes(path_segments_file)

    # Languages Array
#    dtype_languages={'names':('language_ID', 'language'),'formats':('S3','S8')}
#    languages_array = np.genfromtxt(path_languages,dtype_languages,skip_header=1)
    # Not mandatory here

    # Enrollments Array
    dtype_enrollments={'names':('modelid', 'segment', 'side'),'formats':('S16','S16','S1')}
    enrollments_array = np.genfromtxt(path_enrollments_file,dtype=dtype_enrollments,skip_header=1)


    ###################################
    # Creating calls table
    calls2 = rf.rec_join('call_id',call_sides_array,calls_array)
    calls = join_duplicate(calls2,subjects_array,'subject_id','sex')


    #fusionning calls and segments
    columns = ['subject_id','sex','language_id','phone_id']
    Segments, Hash_Seg_Phone = fusion_and_hash_segs_calls(segments_array, calls, columns, 'call_id')

    # Creation of the model table
    Enrollments_dict = create_enrollments_dict(path_enrollments_file)
    Models, Hash_Model_Phone = create_model_table(Enrollments_dict,Segments)

    # Creation of the trials and reference tables
    reference_formats = tuple([get_dtype(Models,'modelid')])\
                        + tuple([get_dtype(Segments,'segment')])\
                        + tuple([get_dtype(enrollments_array,'side')])\
                        + tuple(['S16'])
    reference_dtype = {'names':('modelid', 'segment', 'side','targettype'),'formats':reference_formats}
    Reference_array = np.genfromtxt(path_reference_file,dtype=reference_dtype,skip_header=1)

    return Segments, Models, Reference_array, Hash_Seg_Phone, Hash_Model_Phone


###############################################################################
# Filtering functions

def compute_masks_of_filters_product(List_values,List_filter_functions,List_additionnals_masks,data,phone_number_issue=False):
    """
    Calculate the list of the masks used to generate the partitions described
    in the SRE16 evaluation plan.
    It computes the cartesian product of all the filters values
    Every mask is represented by a boolean array.

    Parameters
    ----------
    List_values : list
        List containing the lists of every values used for each filters.
    List_filter_functions : list of <function>
        List containing the functions reprensenting each filters.
    List_additionnals_masks : list
        For any other type of filter which is not a filter based only on a
        value criteria, as relationnal filters.
        This contains the masks pre-computed and add them to the set of mask.
        Format : [[Mask_A_1,Mask_A_2],[Mask_B_1,...],...]
    data : dict
        dictionnary defined as {'Segments':Segments,
                                'Models':Models,
                                'Reference':Reference_array}
    phone_number_issue : Boolean
        See following note

    Notes
    -----
    * For the developpement data, in one case, for the filtering on the
    phone number, when the phone number is the same for the enrollement segment
    and for the test segments, it remains only the trials which are targets.
    We handle this specific case, we add to the mask 'same number' the nontarget
    trials of the 'different number' mask.
    """
    from itertools import product
    List_masks = []
    # Computing each masks for every filter and store them in the same type
    # of structure than the `List_values`.
    for values, Filter in zip(List_values, List_filter_functions):
        filter_masks = []
        for value in values:
            filter_masks += [Filter(value,data)]
        List_masks += [filter_masks]

    # Same phone_number issue hack
    # Assuming that :
    # * phone number masks are in List_additionnals_masks[0]
    # * phone masks values are ['same','different']
    if phone_number_issue == True:
        mask_same_phone_number = List_additionnals_masks[0][0]
        mask_diff_phone_number = List_additionnals_masks[0][1]
        mask_nontarget_diff_phone_number = np.logical_and(data['Reference']['targettype'] == b'nontarget',mask_diff_phone_number)
        List_additionnals_masks[0][0] = np.logical_or(mask_same_phone_number,mask_nontarget_diff_phone_number) # mask correction

    # Adding the additionnals masks.
    if List_additionnals_masks:
        List_masks += List_additionnals_masks

    # Compute the cartesian product of every set of masks
    final_masks = []
    masks_product = list(product(*List_masks))
    for masks in masks_product:
#        finals_mask += [np.logical_and.reduce(masks)] # explicit version
        final_masks += [np.all(masks,axis=0)] # short version

    return final_masks

def create_reference_labels(reference):
    """ Returns the array of targettypes converted in 0 and 1 as :
        ['target', 'nontarget', 'nontarget',...]
        >> [1, 0, 0,...]
    """
    lbls = np.copy(reference['targettype'])
    lbls[lbls == b'target'] = 1
    lbls[lbls == b'nontarget'] = 0
    lbls = lbls.astype('int')
    return lbls

def beta_labelling(LLR,ln_beta):
    """ apply a filter to the LLR array :
            if the LLR is above log_beta : return 1
            else : return 0
    """
    beta_func = np.vectorize(lambda a,b: 1 if a>b else 0)
    return beta_func(LLR,ln_beta)


def actual_detection_cost(LLR,lbls,number_of_nontarget,number_of_target,nb_beta,beta,ln_beta):
    """
    Compute the actual detection cost

    Parameters
    ----------
    LLR : ndarray
        array containing log-likelyhood ratios
    lbls : ndarray
        array containing targettypes labels represented by ints.
        {0 : nontarget, 1 : target}
    number_of_nontarget, number_of_target : int
        Non targets and target numbers
    nb_beta : int
        number of beta values used.
    beta : ndarray
        array with values of beta
    ln_beta : ndarray
        np.log(beta)
    """
    number_of_Miss = np.zeros((nb_beta,))
    number_of_FalseAlarm = np.zeros((nb_beta,))
#    eps = np.finfo(float).eps

    # Comparison trials and reference
    LLR_2 = np.tile(LLR,(nb_beta,1))
    for i in range(nb_beta):
        LLR_2[i] = beta_labelling(LLR_2[i],ln_beta[i])

    # Compute number of Miss and FalseAlarm
    for i in range(nb_beta):
        for j in range(LLR.size):
            if lbls[j] == 1 and LLR_2[i][j] == 0:
                number_of_Miss[i] += 1
            elif lbls[j] == 0 and LLR_2[i][j] == 1:
                number_of_FalseAlarm[i] += 1

    # Compute probabilities
    P_Miss_Target = number_of_Miss/(number_of_target) if number_of_target>0 else np.nan
    P_FalseAlarm_NonTarget = number_of_FalseAlarm/(number_of_nontarget) if number_of_nontarget>0 else np.nan

    # display variables
#    print("number_of_target = {}".format(number_of_target))
#    print("number_of_nontarget = {}".format(number_of_nontarget))
#    print("number_of_Miss = {}".format(number_of_Miss))
#    print("number_of_FalseAlarm = {}".format(number_of_FalseAlarm))
#    print("P_Miss_Target = {}".format(P_Miss_Target))
#    print("P_FalseAlarm_NonTarget = {}".format(P_FalseAlarm_NonTarget))

    # Compute C_Norm
    C_Norm = P_Miss_Target + beta*P_FalseAlarm_NonTarget

    # Compute C_Primary
    C_Primary = np.mean(C_Norm)

    return C_Primary,P_Miss_Target,P_FalseAlarm_NonTarget


###############################################################################
# Partitions Scoring

def filtered_scoring(List_masks,List_values,Trials_Output,Reference,nb_beta,beta,ln_beta):
    """ Apply actual_detection_cost to all the partitions and
        compute the mean of scores

        Returns also an np.array with the maximun number of targettypes over all partitions :
        [max_number_of_targets,max_number_of_nontargets]
        (used for the equalization)

        Parameters
        ----------
        List_masks : Result of compute_masks_of_filters_product()
        Trials_Output : Recarray of the system output containing `LLR` field
        Reference : Recarray of the reference file containing `targettypes`
        Others parameters are the same as the actual_detection_cost() function
    """

#    from itertools import product
    n_partitions = len(List_masks)
    Score_list = np.zeros(n_partitions)
    Target_types_counter = np.zeros((n_partitions,2))
#    Partitions_values = list(product(*List_values))
    P_Miss_Target_array = np.zeros((n_partitions,2))
    P_FalseAlarm_NonTarget_array = np.zeros((n_partitions,2))

    for i,mask in enumerate(List_masks):
        trial_output = Trials_Output[mask]
        reference = Reference[mask]
        LLR = trial_output['LLR']
        labels = create_reference_labels(reference)
        number_of_nontarget, number_of_target = np.bincount(labels)
        Target_types_counter[i] = np.array([number_of_target,number_of_nontarget])
#        print("\nPartition : {0} {1} {2} {3}".format(*Partitions_values[i]))
        Score_list[i],P_Miss_Target_array[i],P_FalseAlarm_NonTarget_array[i] = actual_detection_cost(LLR,labels,number_of_nontarget,number_of_target,beta.size,beta,ln_beta)

    P_M_avg = np.nanmean(P_Miss_Target_array,axis=0)
    P_FA_avg = np.nanmean(P_FalseAlarm_NonTarget_array,axis=0)
    C_Norm = P_M_avg + beta * P_FA_avg
    
    C_Primary_1 = np.mean(C_Norm)
#    C_Primary_2 = np.mean(Score_list)
    
    return C_Primary_1, np.amax(Target_types_counter,axis=0)
