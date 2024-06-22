import pandas as pd

def troublesome_尻ㄙㄟ(df, knowledge_data):
    # st.write(df)
    # st.write(df.T[knowledge_data['phn'].to_list()].T)
    df1 = df.T[knowledge_data['phn'].to_list()].T
    df2 = df1.T
    # st.write(df2.T)
    representative_phn_of_units = (df1.idxmax(axis=0))
    max_prob_of_representative_phns = df1.max(axis=0)
    representative_phn_of_units_idx = representative_phn_of_units.apply(lambda x: knowledge_data['phn'].to_list().index(x))
    # df_sortedunits = df2.T.sort_values(by=representative_phn_of_units_idx, axis=1, ascending=True)
    # st.write(representative_phn_of_units)
    # st.write(max_prob_of_representative_phns)
    # st.write(representative_phn_of_units_idx)
    df_sort_by_aligned_phn_and_prob = pd.concat([representative_phn_of_units, max_prob_of_representative_phns, 
            representative_phn_of_units_idx                                    ], axis=1)
    df_sort_by_aligned_phn_and_prob.columns = ['phn', 'prob', 'phn_idx']
    df_sort_by_aligned_phn_and_prob['sorter_value'] = (representative_phn_of_units_idx.max()+1-df_sort_by_aligned_phn_and_prob['phn_idx'] + max_prob_of_representative_phns / 100)
    df_sort_by_aligned_phn_and_prob = df_sort_by_aligned_phn_and_prob.sort_values(by='sorter_value', ascending=False)
    # st.write(df_sort_by_aligned_phn_and_prob)
    dfcool = df1.loc[:, df_sort_by_aligned_phn_and_prob.index]
    # plot_heatmap(dfcool)
    # st.write(df1.loc[:, df_sort_by_aligned_phn_and_prob.index])

    grouings_good = df2.idxmax(axis=1).value_counts()
    # knowledge_data[['phn', 'cls']].groupby('phn').count()
    phn_to_cls = knowledge_data[['phn', 'cls']].groupby('phn').first()
    # grouings_good group by cls
    grouings_good_cls = grouings_good.to_frame()
    grouings_good_cls['cls1'] = grouings_good_cls.index.map(lambda x: phn_to_cls.loc[x, 'cls'])
    grouings_good_cls = grouings_good_cls.reset_index()
    grouings_good_cls['phn'] = grouings_good_cls['index']
    grouings_good_cls = grouings_good_cls.drop(columns=['index'])
    grouings_good_cls = grouings_good_cls[[
        'cls1',
        'phn',
        'count',
    
    ]]
    # grouings_good_cls = grouings_good_cls.join(phn_to_cls, on='index')
    # sum 'count' by 'cls1'

    grouings_good_cls = grouings_good_cls.groupby('cls1').aggregate('sum')
    # if 0, add 0 to all cls
    for it in knowledge_data['cls'].unique():
        if it not in grouings_good_cls.index:
            grouings_good_cls.loc[it] = 0
    # st.write(grouings_good_cls)
    mycls = (
        knowledge_data['cls'].unique().tolist()
    )
    grouings_good_cls = (grouings_good_cls['count'])
    resultgrouping = grouings_good_cls.loc[mycls].values.tolist()

    return dfcool, resultgrouping

def entropy欸尻ㄙㄟ(df, entropy_phn, entropy_unit, knowledge_data):

    df['entropy_phn'] = entropy_phn
    df = df.sort_values(by='entropy_phn', ascending=False)
    df = df.drop(columns=['entropy_phn'])

    df = df.T
    df['entropy_unit'] = entropy_unit
    df = df.sort_values(by='entropy_unit', ascending=False)
    df = df.drop(columns=['entropy_unit'])
    df = df.T

    df1 = df.T[knowledge_data['phn'].to_list()].T
    return df1.iloc[::1, ::-1], df.iloc[::-1, ::-1]
