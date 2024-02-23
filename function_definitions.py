import re
import dask
dask.config.set(scheduler="processes")
import requests
import requests_random_user_agent
from bs4 import BeautifulSoup
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
import pandas as pd
import numpy as np
from nltk.corpus import words
import time
from nltk.stem import PorterStemmer
from wordfreq import top_n_list
import statsmodels.api as sm
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from numba import jit
from scipy.stats import f


#function that downloads the idx files for a type of file from the SEC for a given year
def get_idx_files(year, path, FILE):
    #find and open the idx file for each quarter
    for quarter in [1,2,3,4]:
        url = 'https://www.sec.gov/Archives/edgar/full-index/%s/QTR%s/master.idx'%(year,quarter)
        response = requests.get(url)
        #some idx files generate an encoding error when openning with response.text
        #can be fixed by first saving the idx files and then
        #replacing error bites when readng the file
        #save master.idx file
        with open(path+'master/master_{}q{}.idx'.format(year,quarter),"wb") as master:
            master.write(response.content)
        #read master.idx file
        with open(path+'master/master_{}q{}.idx'.format(year,quarter),"r", errors="replace") as master:
            master_idx = master.read()
            #save all 10-K disclosures from the idx file
        with open(path+'{}/idx_{}_{}.txt'.format(FILE, FILE, year), "a") as f:
        #only save the relevant fillings
            pattern = re.compile(r'\d+\|.*\|10-K\|.*\|edgar/data/.*.txt')
            for line in pattern.findall(master_idx):
                    f.write(line)
                    f.write("\n")
            

#function that finds the url, filling date and company name in a line from the SEC's index
@dask.delayed()
def get_url(line):
    url_found = {}
    
    CIK_pattern = re.compile('(\d*)|.*\|10-K\|')
    CIK = CIK_pattern.findall(line)[0]
    
    url_pattern = re.compile('10-K\|.*\|(edgar/data/.*.txt)'.format(CIK))
    url = url_pattern.findall(line)[0]
    url_found[CIK] = ['https://www.sec.gov/Archives/'+url]
    
    filling_date_pattern = re.compile('10-K\|(.*)\|edgar/data'.format(CIK))
    filling_date = filling_date_pattern.findall(line)[0]
    url_found[CIK].append(filling_date)
    
    company_name_pattern = re.compile('{}\|(.*)\|10-K'.format(CIK))
    company_name = company_name_pattern.findall(line)[0]
    url_found[CIK].append(company_name)
    
    return url_found


#function that gets the descritions of cyber tactics from the MITRE ATT&CK website
#url should be one of the MITRE tactics' on the MITRE website 
def read_cyber_description(url):
    #read in the page and clean the text
    response = requests.get(url)
    text = BeautifulSoup(response.text, 'lxml').get_text()
    text = [re.sub('\s+',' ',x) for x in text.split('\n') if re.sub('\s+','',x)]
    text = ' '.join(text)
    text = re.sub('load more results', '', text) 
    #find the start of each technique description
    technique_pattern = re.compile('T\d+')
    sub_technique_pattern = re.compile('[^\d]\.\d+')
    techniques = [start_pattern.start() for start_pattern in technique_pattern.finditer(text)]
    
    #find all descriptions of techniques and sub techniques
    descritions = {}
    for i in range(len(techniques)-1):
        txt = text[techniques[i]:techniques[i+1]].split('  ')
        if (len(txt)<5):
            descritions[txt[1].strip()] = [txt[2].strip()]
            continue
        sub_descritions = {}
        for j in range((len(txt)-1)//3):
            sub_descritions[txt[j*3+1].strip()] = [txt[j*3+2].strip()]
        descritions[txt[1].strip()] = sub_descritions
        
    txt = text[techniques[i+1]:].split('  ')
    if (len(txt)<5):
        descritions[txt[2].strip()] = [txt[3].strip()]
    else:
        sub_descritions = {}
        for j in range((len(txt)-1)//3):
            sub_descritions[txt[j*3+1].strip()] = [txt[j*3+2].strip()]
        descritions[txt[1].strip()] = sub_descritions
    
    #format the descriptions as a dataframe
    descritions_df = []
    for descrition in descritions.keys():
        try:
            idx = pd.MultiIndex.from_product([[descrition],descritions[descrition].keys()],
                                             names = ['technique', 'sub_technique'])
        except:
            idx = pd.MultiIndex.from_product([[descrition],[descrition]], names = ['technique', 'sub_technique'])
        df = pd.DataFrame(descritions[descrition], index = ['Description']).T
        df.index = idx
        descritions_df.append(df)
    descritions_df = pd.concat(descritions_df)
    
    return descritions_df


#function that takes text as input and returns a list of tokennized sentences
#text is the input text,
#ticker is either the ticker of the firm whose 10-K the text is from or None,
#raw is true if the text is unprosessed from a 10-k,
#merge is True if you want to merge consecutive sentences (is the case for 10-Ks)
#sentences is should only be false if the text is one sentence
#find_item_1A_ is True if you want to also identify the limits of item 1A
#sleep adds 0.5s of sleep (used when doing calling the function many times with raw data) 
def get_tokens(text, ticker = None, raw = True, merge = True, sentences = True, find_item1A_ = False, sleep = True):
    #if the input is raw, extract the text
    if raw:
        if find_item1A_:
            document, item1A = get_text(text, find_item1A_ = True)
            if item1A != 'ITEM 1A NOT FOUND':
                #delete extra spaces and short sentences
                item1A = re.sub('\n',' ', item1A)
                item1A = re.sub(r'&nbsp', ' ', item1A)
                item1A = re.sub(r'\xa0;', ' ', item1A)
                item1A = re.sub(r'&#\d+;', ' ', item1A)
                item1A = re.sub('<.*?>', ' ', item1A)
                item1A = re.sub('\s{2,}',' ', item1A)
                lines_1A = sent_tokenize(item1A)
                lines_1A = [line for line in lines_1A if len(word_tokenize(line)) > 15]
        else:
            document = get_text(text)
        #delete extra spaces and short sentences
        document = re.sub('\n',' ', document)
        document = re.sub(r'&nbsp', ' ', document)
        document = re.sub(r'\xa0;', ' ', document)
        document = re.sub(r'&#\d+;', ' ', document)
        document = re.sub('<.*?>', ' ', document)
        document = re.sub('\s{2,}',' ', document)
        lines = sent_tokenize(document)
        lines = [line for line in lines if len(word_tokenize(line)) > 15]
        
        if find_item1A_:
            if item1A != 'ITEM 1A NOT FOUND':
                try:
                    indices_lines = [i for i, x in enumerate(lines) if x in lines_1A]
                    indices_lines_1A = [lines_1A.index(x) for x in lines if x in lines_1A]
                    
                    first_1A = indices_lines[np.argmin(indices_lines_1A)]
                    indices_lines_1A = indices_lines_1A[np.argmin(indices_lines_1A):]
                    last_1A = indices_lines[np.argmax(indices_lines_1A)]
                    
                    if last_1A < first_1A:
                        first_1A = np.nan
                        last_1A = np.nan
                except:
                    first_1A = np.nan
                    last_1A = np.nan
            else:
                first_1A = np.nan
                last_1A = np.nan
    else:
        if sentences:
            lines = sent_tokenize(text)
        else:
            lines = [text]
    
    #tokenize each line
    if ticker:
        tokens_per_line = {}
        tokens_per_line[ticker] = []
    else:
        tokens_per_line = []
        
    common_words = top_n_list('en', 100)
    not_alphabet = re.compile('[^a-z]') # = anything that's not a letter
    stop_words = stopwords.words('english')
    
    temp = []
    for line in lines:
        tokens = gensim.utils.simple_preprocess(line) #lowercases and tokenizes
        #delete common words and stopwords
        tokens = [word for word in tokens if word not in common_words]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [not_alphabet.sub('',word) for word in tokens]
        #tokens = stem(tokens)
        if(len(tokens)):
            temp.append(tokens)

    #merge consecutive sentences to have paragraphs about the same length as the techniques and tactics            
    if merge:
        if find_item1A_:
            merged, merge_list = merge_sentences(temp, ret_merge_list = True)
        else:
            merged = merge_sentences(temp)
        if merged:
            if ticker:
                tokens_per_line[ticker] = merged
            else:
                tokens_per_line = merged 
            if find_item1A_ and not np.isnan(first_1A) and not np.isnan(last_1A):
                first_1A = first_1A - sum(merge_list[:first_1A+1])
                last_1A = last_1A - sum(merge_list[:last_1A+1])
        else:
            if ticker:
                tokens_per_line[ticker] = temp
            else:
                tokens_per_line = temp
    else:
        if ticker:
            tokens_per_line[ticker] = temp
        else:
            tokens_per_line = temp
    
    if sleep:
        time.sleep(0.5)
    
    if find_item1A_:
        if ticker:
            idx_1A = {}
            idx_1A[ticker] = [first_1A, last_1A]
        else:
            idx_1A = [first_1A, last_1A]
        return idx_1A
    else:
        return tokens_per_line


#function that takes a list of sentences and merges them to have paragraphs with a length
#close to the desired length
#sentences should be a list of sentences
#ret_merge_list if True, returns a mapping of sentences that were merged 
def merge_sentences(sentences, ret_merge_list = False):
    desired_len = 40 # = avg len of techniques and tactics
    avg_line_len = np.mean([len(t) for t in sentences], dtype = int)
    #if avg len is shorter than desired len, merge sentences
    if (desired_len/avg_line_len > 1):
        merge_list = [0]
        merged = []
        merged.append(sentences[0])
        i = 1
        j = 0
        while i < len(sentences):
            #if get further from the desired length by merging, stop merging onto current paragraph and start a new one
            len_after_merge = len(merged[j])+len(sentences[i])
            if abs(len_after_merge-desired_len) > abs(len(merged[j])-desired_len):
                merged.append(sentences[i])
                j += 1
                i += 1
                merge_list.append(0)
                continue
            #otherwise merge the sentence on the current paragraph
            merged[j].extend(sentences[i])
            merge_list.append(1)
            i += 1
        if ret_merge_list:
            return merged, merge_list
        else:
            return merged
    else:
        if ret_merge_list:
            return None, None
        else:
            return None

    
#function that takes as input raw text from a 10-k statement and returns the cleaned text
#find_item_1A_ also returns item 1A from the text if True
def get_text(raw_text, find_item1A_ = False):
    #Regex to find <DOCUMENT> tags
    document_start_pattern = re.compile(r'<DOCUMENT>')
    document_end_pattern = re.compile(r'</DOCUMENT>')
    
    # Regex to find <TYPE> tag preceeding any characters until the end of the line
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    document_starts = [start_pattern.end() for start_pattern in document_start_pattern.finditer(raw_text)]
    document_ends = [end_pattern.start() for end_pattern in document_end_pattern.finditer(raw_text)]
    
    # find the type of each part of the raw text file
    document_types = [type_[len('<TYPE>'):] for type_ in type_pattern.findall(raw_text)]
    
    # Loop through the sections of the document and save the 10-k part
    document = []
    for document_type, document_start, document_end in zip(document_types, document_starts, document_ends):
        if document_type == '10-K' or document_type == '10-K/A':
            #only retain the first occurance
            if not len(document):
                document = raw_text[document_start:document_end]
    
    if find_item1A_:
        item1A_raw = find_item1A(document)
        soup = BeautifulSoup(item1A_raw, "html.parser")
        # delete all script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        item1A = soup.get_text()
        
    soup = BeautifulSoup(document, "html.parser")
    # delete all script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    document = soup.get_text()
    if find_item1A_:
        return document, item1A
    else:   
        return document


#fucntion that identifies item 1A from a raw 10-K
def find_item1A(document):
    #we are interested in Item 1A risk factors
    #in the raw txt file it is either Item 1A, Item&#1601A,...
    #Also find Item 1B, Item 2, Item 3, Item 4 and Item 5 to know where 1A ends
    pattern = re.compile(r'(((>|\n)(|\s+)(i|I)(|(<[^>]+>)+)(t|T)(|(<[^>]+>)+)(e|E)(|(<[^>]+>)+)(m|M))'
                           '(\s+|&#160;|&nbsp;| </b><b>|(<[^>]+>)+|&#xA0;|&#xa0;)(|&nbsp;|&#xa0;|&#xA0;|&#160;|(<[^>]+>)+|\s+)'
                           '(|(<[^>]+>)+)(1(|(<[^>]+>)+)(|\s+)(A|a)|1(|(<[^>]+>)+)(|\s+)(B|b)|2|3|4|5))')
    
    # Use finditer to match the regex
    matches = pattern.finditer(document)

    # Create a dataframe with start and end of each instance of Item 1A and Item 1B found
    df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])
    try:
        df.columns = ['item', 'start', 'end']
    except:
        return 'ITEM 1A NOT FOUND'
    df['item'] = df.item.str.lower()
    
    # Get rid of unnesesary characters from the dataframe
    df.replace('&#160;',' ',regex=True,inplace=True)
    df.replace('&nbsp;',' ',regex=True,inplace=True)
    df.replace('&#xa0;',' ',regex=True,inplace=True)
    #df.replace('</b><b>','',regex=True,inplace=True)
    #df.replace('>tem','item',regex = True,inplace=True)
    #df.replace('>em','item',regex = True,inplace=True)
    df.replace('<[^>]+>','',regex = True,inplace = True)
    df.replace(' ','',regex=True,inplace=True)
    df.replace('\.','',regex=True,inplace=True)
    df.replace('>','',regex=True,inplace=True)
    df.replace('\n','',regex=True,inplace=True)
    
    #if no instance of item 1a found
    if(not len(df[df.item == 'item1a'])):
        return 'ITEM 1A NOT FOUND'
    
    #if several instances of item 1a are found, remove the ones that are after all the other items
    #since won't know the end of item 1a in that case
    if(len(df[df.item == 'item1a']) >= 2):
        while((df[df.start == df.start.max()].item == 'item1a').values and (len(df[df.item == 'item1a']) >= 2)):
            df = df[df.start != df.start.max()]

    # Drop duplicates keeping the last 
    df.sort_values('start', ascending = True, inplace = True)
    df_unique = df.drop_duplicates(subset=['item'], keep='last').copy()
    
    #find the item that is right after item 1a
    next_item = find_next_item(df_unique)
         
    if(next_item == 'NOT FOUND'):
        return 'ITEM 1A NOT FOUND'
    
    # Set item as the dataframe index
    df_unique.set_index('item', inplace=True)
    
    item_1a_raw, correct_item_found = extract_item_1a(df_unique, document, next_item)
    
    #if the the text identified does not seem to be correct
    #we can test and see whether the previous instance that was found is the right one
    if not correct_item_found:
        if (len(df[df.item == 'item1a']) > 1):
            #delete the bounds that were previously considered
            df = pd.merge(df,df_unique, how = 'outer', indicator=True)
            df = df[df._merge != 'both'].drop('_merge', axis=1).reset_index(drop=True)
            
            #if item 1a is now the last element, add back the previously considered bounds for the other items
            if((df[df.start == df.start.max()].item == 'item1a').values):
                df_unique = df_unique.reset_index()
                df_unique = df_unique[df_unique.item != 'item1a']
                df = pd.merge(df, df_unique, how = 'outer')

            #re-exctract item 1a using the new bounds
            df_unique = df.drop_duplicates(subset=['item'], keep='last').copy()
            #find the item that is right after item 1a
            next_item = find_next_item(df_unique)

            if(next_item == 'NOT FOUND'):
                return 'ITEM 1A NOT FOUND'

            # Set item as the dataframe index
            df_unique.set_index('item', inplace=True)

            item_1a_raw, correct_item_found = extract_item_1a(df_unique, document, next_item)

        if not correct_item_found:
            return 'ITEM 1A NOT FOUND'
    
    #remove leading and trailing whitespaces
    item_1a_raw = item_1a_raw.strip()
    
    return item_1a_raw


#function that returns the vector representation of the input tokens using a trained model
#model should be a trained gensim doc2vec model
#tokens should be a list of words
#tag is a string associated to the tokens
#if words is true also return the words from the Tagged doc2vec document
def get_vect(model, tokens, tag, words = False):
    sentence = gensim.models.doc2vec.TaggedDocument(tokens, [tag])
    model.random.seed(0)
    if words:
        return model.infer_vector(sentence.words), sentence.words
    return model.infer_vector(sentence.words)


#function that takes a permco and a start and end date and returns monthly returns for that permco for 
#all available dates between start and end date
def get_returns(permco, db, start_date, end_date, ticker):
    Request = """select date, ret, shrout, prc, hsiccd from crsp.msf where permco in ({})
              and date >= '{}' and date <='{}'""".format(permco, start_date, end_date)
    data = db.raw_sql(Request)
    data.set_index('date', inplace = True)
    data.index = pd.to_datetime(data.index)
    data.index = data.index.to_period('M')
    #there are months with several observations for some reason
    data = data[~data.index.duplicated(keep='first')]
    #calculate market cap
    data['Market Cap'] = data['shrout']*1000*data['prc'].abs() #shrout in thousands
    data.drop(columns = ['shrout', 'prc'], inplace = True)
    #drop returns that are missing (==-66, -77, -88 or -99)
    missing_returns = [-66, -77, -88, -99]
    data = data[~np.isin(data.ret, missing_returns)]
    #replace missing values from the industry classifications with NaN
    data[data.hsiccd == 0] = np.nan
    
    data.columns = [ticker, ticker, ticker]
    #return monthly returns, monthly market cap and Standard Industrial classification code
    return data.iloc[:,0], data.iloc[:,1], data.iloc[:,2]


#function that returns the stemmed version of the input words
def stem(words):
    ps = PorterStemmer()
    stemmed = []
    for w in words:
        stemmed.append(ps.stem(w))
    return stemmed


#function that takes as input the paragraph vector of a test paragraph and paragraph vector of the cybersecurity
#tactic descriptions from mitre and returns the cosine similarity of the test paragraph to each tactic
#@dask.delayed()
def get_sim(test_vector, tactic_vectors):
    similarity = []
    for tactic_vector in tactic_vectors:
        similarity.append(max(cosine_similarity([tactic_vector],[test_vector])[0][0],0))# sim = max(cosine_sim, 0)
    return similarity


#function that matches the Standard Industrial Classification to the Fama-French 12 industry classification
def match_FF_industry(SIC, SIC_to_FF):
    if np.isnan(SIC):
        return np.nan
    try:
        return SIC_to_FF[(SIC_to_FF.SIC_low <= SIC) & (SIC_to_FF.SIC_high >= SIC)].FF_12.values[0]
    except:
        #others = 12
        return 12

    
#function that reads in tokens from a csv file and returns the number of paragraphs
#and the avreage length of the paragraphs
@dask.delayed()
def get_token_stats(path):
    tokens = pd.read_csv(path, index_col = 0, dtype = 'object').T
    nb_paragraphs = len(tokens.columns)
    avg_paragraph_len = tokens.count().mean()
    return [nb_paragraphs, avg_paragraph_len]


#function to calculate the cosine similarity between two vectors u and v
#uses numba to speed up the calculations
@jit(nopython = True)
def cos_sim(u,v):
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


#function that computes the cosine similarity (only positive part) between a test vector and the tactic vectors
#uses numba for speed (for 1 firm, get_sim takes around 2 minutes, fast_sim takes around a second)
def fast_sim(test_vector, tactic_vectors):
    similarity = []
    test_vector = np.array(test_vector)
    tactic_vectors = np.array(tactic_vectors)
    for tactic_vector in tactic_vectors:
        similarity.append(max(cos_sim(tactic_vector,test_vector),0))
    return similarity


#function that computes the cosine similarity between a test vector and the tactic vectors
#uses numba for speed (for 1 firm, get_sim takes around 2 minutes, fast_sim takes around a second)
def fast_sim_neg(test_vector, tactic_vectors):
    similarity = []
    test_vector = np.array(test_vector)
    tactic_vectors = np.array(tactic_vectors)
    for tactic_vector in tactic_vectors:
        similarity.append(cos_sim(tactic_vector,test_vector)) #allows for negative similarity
    return similarity


#function that calculates the alphas with respect to factor models (quantile portfolios)
#quintile returns should be a datdaframe containing a time series of returns
#FF5 shoud be a dataframe containing the time series of returns of the FF5 factors
def get_alphas(quintile_returns, FF5):
    table = pd.DataFrame(np.ones([4,6]), index = ['Excess return', 'CAPM alpha', 'FFC alpha','FF5 alpha'],
                    columns = ['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)', 'Q5-Q1'])*np.nan
    pval_table = table.copy()
    t_stat_table = table.copy()
    #Sharpe, Treynor and Sortino ratio
    Ratio_table = pd.DataFrame(np.ones([3,6]), index = ['Annualized Sharpe Ratio','Annualized Treynor Ratio',
                                                     'Annualized Sortino Ratio'],
                    columns = ['Q1 (low)', 'Q2', 'Q3', 'Q4', 'Q5 (high)', 'Q5-Q1'])*np.nan
    idx = quintile_returns.index
    FFC_factors = ['Mkt-RF','HML','SMB','UMD']
    FF5_factors = ['Mkt-RF','HML','SMB','RMW', 'CMA']
    
    excess_ret = []
    for q in range(5):
        #excess returns
        er = quintile_returns[f'quintile_{q}'].sub(FF5.loc[idx,'RF'],axis = 0)
        excess_ret.append(er)
        table.iloc[0,q] = er.mean()
        pval_table.iloc[0,q] = sm.OLS(er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const 
        t_stat_table.iloc[0,q] = sm.OLS(er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).tvalues.const
        Ratio_table.iloc[0,q] = er.mean()/er.std() * np.sqrt(12)
        downside_deviation = np.sqrt((er[er<0]**2).sum()/er.shape[0])
        Ratio_table.iloc[2,q] = er.mean()/downside_deviation * np.sqrt(12)
        #CAPM alpha
        capm = sm.OLS(er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        table.iloc[1,q] = capm.params.const
        pval_table.iloc[1,q] = capm.pvalues.const
        t_stat_table.iloc[1,q] = capm.tvalues.const
        Ratio_table.iloc[1,q] = er.mean()/capm.params['Mkt-RF'] * np.sqrt(12)
        #FFC alpha
        ffc = sm.OLS(er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        table.iloc[2,q] = ffc.params.const
        pval_table.iloc[2,q] = ffc.pvalues.const
        t_stat_table.iloc[2,q] = ffc.tvalues.const
        #FF5 alpha
        ff5 = sm.OLS(er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        table.iloc[3,q] = ff5.params.const
        pval_table.iloc[3,q] = ff5.pvalues.const
        t_stat_table.iloc[3,q] = ff5.tvalues.const
    
    #long-short portfolio
    #excess returns
    ls_er = excess_ret[-1].sub(excess_ret[0].values)
    table.iloc[0,5] = ls_er.mean()
    pval_table.iloc[0,5] = sm.OLS(ls_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const
    t_stat_table.iloc[0,5] = sm.OLS(ls_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).tvalues.const
    Ratio_table.iloc[0,5] = ls_er.mean()/ls_er.std() * np.sqrt(12)
    downside_deviation = np.sqrt((ls_er[ls_er<0]**2).sum()/ls_er.shape[0])
    Ratio_table.iloc[2,5] = er.mean()/downside_deviation * np.sqrt(12)
    #CAPM alpha
    capm = sm.OLS(ls_er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[1,5] = capm.params.const
    pval_table.iloc[1,5] = capm.pvalues.const
    t_stat_table.iloc[1,5] = capm.tvalues.const
    Ratio_table.iloc[1,5] = er.mean()/capm.params['Mkt-RF'] * np.sqrt(12)
    #FFC alpha
    ffc = sm.OLS(ls_er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[2,5] = ffc.params.const
    pval_table.iloc[2,5] = ffc.pvalues.const
    t_stat_table.iloc[2,5] = ffc.tvalues.const
    #FF5 alpha
    ff5 = sm.OLS(ls_er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[3,5] = ff5.params.const
    pval_table.iloc[3,5] = ff5.pvalues.const
    t_stat_table.iloc[3,5] = ff5.tvalues.const
    
    return table, pval_table, t_stat_table, Ratio_table


# function that implements the Gibbons-Ross-Shanken test
#alphas should be a list of alphas,
#residuals should be a dataframe of residuals
#factors should a dataframe of the factors with nb_columns = nb_factors,
#if for_barillas is True, returns the GRS statistic without multiplying with the constant
def GRS(alphas, residuals, factors, for_barillas = False):
    T = residuals.shape[0] # number of time-series observations
    N = len(alphas) # number of portfolios
    K = factors.shape[1] # number of factors
    
    sigma_hat = residuals.cov() # residual covariance matrix
    sigma_hat_inv = np.linalg.pinv(sigma_hat)
    omega_hat = factors.cov() # factor covariance matrix
    omega_hat_inv = np.linalg.pinv(omega_hat)
    
    mu = factors.mean()
    
    if not for_barillas:
        GRS = ((T-N-K)/N)*((alphas @ sigma_hat_inv @ alphas)/(1 + mu @ omega_hat_inv @ mu))
    else:
        GRS = (alphas @ sigma_hat_inv @ alphas)/(1 + mu @ omega_hat_inv @ mu)
    
    p_value = 1 - f.cdf (GRS, N, T-N-K)
    return GRS, p_value


#function that transforms a list into a dataframe and standardizes it
#used in Fama-Macbeth regressions
def standardize(input_):
    df = pd.DataFrame(input_)
    df = (df-np.mean(df.values))/np.std(df.values)
    return df


#function that computes the t-stat and p value of a risk premium
#used in Fama-Macbeth regressions
def compute_stats(results):
    model_ = sm.OLS(results, np.ones(len(results))).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    return model_.tvalues[0],model_.pvalues[0]


#function that computes cumulative returns from simple returns
def cumulate_returns(group):
    result = (group + 1).prod() - 1
    return result


#function that calculates the alphas with respect to factor models (tercile portfolios)
def get_alphas_(quintile_returns, FF5):
    table = pd.DataFrame(np.ones([4,4]), index = ['Excess return', 'CAPM alpha', 'FFC alpha','FF5 alpha'],
                    columns = ['Q1 (low)', 'Q2', 'Q3 (high)', 'Q3-Q1'])*np.nan
    pval_table = table.copy()
    t_stat_table = table.copy()
    #Sharpe, Treynor and Sortino ratio
    Ratio_table = pd.DataFrame(np.ones([3,4]), index = ['Annualized Sharpe Ratio','Annualized Treynor Ratio',
                                                     'Annualized Sortino Ratio'],
                    columns = ['Q1 (low)', 'Q2', 'Q3 (high)', 'Q3-Q1'])*np.nan
    idx = quintile_returns.index
    FFC_factors = ['Mkt-RF','HML','SMB','UMD']
    FF5_factors = ['Mkt-RF','HML','SMB','RMW', 'CMA']
    
    excess_ret = []
    for q in range(3):
        #excess returns
        er = quintile_returns[f'quintile_{q}'].sub(FF5.loc[idx,'RF'],axis = 0)
        excess_ret.append(er)
        table.iloc[0,q] = er.mean()
        pval_table.iloc[0,q] = sm.OLS(er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const
        t_stat_table.iloc[0,q] = sm.OLS(er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).tvalues.const
        Ratio_table.iloc[0,q] = er.mean()/er.std() * np.sqrt(12)
        downside_deviation = np.sqrt((er[er<0]**2).sum()/er.shape[0])
        Ratio_table.iloc[2,q] = er.mean()/downside_deviation * np.sqrt(12)
        #CAPM alpha
        capm = sm.OLS(er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        table.iloc[1,q] = capm.params.const
        pval_table.iloc[1,q] = capm.pvalues.const
        t_stat_table.iloc[1,q] = capm.tvalues.const
        Ratio_table.iloc[1,q] = er.mean()/capm.params['Mkt-RF'] * np.sqrt(12)
        #FFC alpha
        ffc = sm.OLS(er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        table.iloc[2,q] = ffc.params.const
        pval_table.iloc[2,q] = ffc.pvalues.const
        t_stat_table.iloc[2,q] = ffc.tvalues.const
        #FF5 alpha
        ff5 = sm.OLS(er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        table.iloc[3,q] = ff5.params.const
        pval_table.iloc[3,q] = ff5.pvalues.const
        t_stat_table.iloc[3,q] = ff5.tvalues.const
    
    #long-short portfolio
    #excess returns
    ls_er = excess_ret[-1].sub(excess_ret[0].values)
    table.iloc[0,3] = ls_er.mean()
    pval_table.iloc[0,3] = sm.OLS(ls_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const
    t_stat_table.iloc[0,3] = sm.OLS(ls_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).tvalues.const
    Ratio_table.iloc[0,3] = ls_er.mean()/ls_er.std() * np.sqrt(12)
    downside_deviation = np.sqrt((ls_er[ls_er<0]**2).sum()/ls_er.shape[0])
    Ratio_table.iloc[2,3] = er.mean()/downside_deviation * np.sqrt(12)
    #CAPM alpha
    capm = sm.OLS(ls_er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[1,3] = capm.params.const
    pval_table.iloc[1,3] = capm.pvalues.const
    t_stat_table.iloc[1,3] = capm.tvalues.const
    Ratio_table.iloc[1,3] = er.mean()/capm.params['Mkt-RF'] * np.sqrt(12)
    #FFC alpha
    ffc = sm.OLS(ls_er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[2,3] = ffc.params.const
    pval_table.iloc[2,3] = ffc.pvalues.const
    t_stat_table.iloc[2,3] = ffc.tvalues.const
    #FF5 alpha
    ff5 = sm.OLS(ls_er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[3,3] = ff5.params.const
    pval_table.iloc[3,3] = ff5.pvalues.const
    t_stat_table.iloc[3,3] = ff5.tvalues.const
    
    return table, pval_table, t_stat_table, Ratio_table

################################################################################################################################
################################################################################################################################
#################################### Functions used in the event study #########################################################
################################################################################################################################
################################################################################################################################



#function that takes a cusip and a start and end date and returns monthly returns for that cusip for 
#all available dates between start and end date
def get_returns_from_cusip(cusip, db, start_date, end_date):
    Request = """select date, ret, shrout, prc from crsp.dsf where cusip in ({})
              and date >= '{}' and date <='{}'""".format("'{}'".format(cusip), start_date, end_date)
    data = db.raw_sql(Request)
    data.set_index('date', inplace = True)
    data.index = pd.to_datetime(data.index, format = '%Y-%m-%d')
    
    #there are months with several observations for some reason
    data = data[~data.index.duplicated(keep='first')]
    #calculate market cap
    data['Market Cap'] = data['shrout']*1000*data['prc'].abs() #shrout in thousands
    data.drop(columns = ['shrout', 'prc'], inplace = True)
    
    #drop returns that are missing (==-66, -77, -88 or -99)
    missing_returns = [-66, -77, -88, -99]
    data = data[~np.isin(data.ret, missing_returns)]
    
    data.columns = [cusip, cusip]
    #return daily returns and daily market cap
    return data.iloc[:,0], data.iloc[:,1]


#function that finds the divisor of y-x that is the closest to the target value
def find_closest_divisor(x, y, target):
    # Calculate the absolute difference between y and x
    diff = abs(y - x)

    # Find all divisors of the difference using list comprehensions
    divisors = [i for i in range(1, diff + 1) if diff % i == 0]

    # Sort the divisors by their absolute difference to the target
    divisors.sort(key=lambda divisor: abs(divisor - target))
    
    # The closest divisor is the first element in the sorted list
    closest_divisor = divisors[0]
    
    return closest_divisor

#function that computes the product of the consecutive non-null elements of a series
def product_of_consecutive_non_nulls(series):
    products = []

    for i in range(len(series) - 2):
        if not any(pd.isna(series.iloc[i:i+3])):
            products.append((series.iloc[i:i+3]+1).product()-1)

    return products


################################################################################################################################
################################################################################################################################
#################################### Functions used for replicating Florackis et al ############################################
################################################################################################################################
################################################################################################################################

#function that finds the url and filling date of a specified file for a given company and a given year among a list of 10-Ks
@dask.delayed()
def get_url_old(ticker, list_10k, FILE, CIK):
    url_found = {}
    
    #get rid of zeros in the front
    CIK = str(int(CIK))
    pattern = re.compile('(?:\n|0){}\|.*\|10-K\|.*\|(edgar/data/.*.txt)'.format(CIK))
    found = pattern.findall(list_10k)
    if(len(found)):
        url_found[ticker]= ['https://www.sec.gov/Archives/'+found[0]]
    else:
        url_found[ticker] = ['URL NOT FOUND']
    
    #find the filling date
    pattern = re.compile('(?:\n|0){}\|.*\|10-K\|(.*)\|edgar/data/.*.txt'.format(CIK))
    found = pattern.findall(list_10k)
    if(len(found)):
        url_found[ticker].append(found[0])
    else:
        url_found[ticker].append('DATE NOT FOUND')
        
    return url_found


#function that finds Item 1A. Risk factors in a 10-K statement
#takes as input the url to the 10-k and the company's ticker and returns the raw version of Item 1A as a string
@dask.delayed()
def find_risk_factors(url, ticker):
    #read in the text from the url
    response = requests.get(url)
    raw_10k = response.text
    
    #Regex to find <DOCUMENT> tags
    document_start_pattern = re.compile(r'<DOCUMENT>')
    document_end_pattern = re.compile(r'</DOCUMENT>')
    # Regex to find <TYPE> tag preceeding any characters until the end of the line
    type_pattern = re.compile(r'<TYPE>[^\n]+')

    #find the start and end of each part of the raw text file (ex 10-K, XML, JSON, ...)
    #the start of the document is right after the document start pattern hence start_pattern.end()
    #the end of the document is right before the document end pattern hence end_pattern.start()
    document_starts = [start_pattern.end() for start_pattern in document_start_pattern.finditer(raw_10k)]
    document_ends = [end_pattern.start() for end_pattern in document_end_pattern.finditer(raw_10k)]

    # find the type of each part of the raw text file
    document_types = [type_[len('<TYPE>'):] for type_ in type_pattern.findall(raw_10k)]
    
    # Loop through the sections of the document and save the 10-k part
    document = []
    for document_type, document_start, document_end in zip(document_types, document_starts, document_ends):
        if document_type == '10-K' or document_type == '10-K/A':
            #only retain the first occurance
            if not len(document):
                document = raw_10k[document_start:document_end]
    
    #lower case all letters
    document = document.lower()
    
    #we are interested in Item 1A risk factors
    #in the raw txt file it is either Item 1A, Item&#1601A,...
    #Also find Item 1B, Item 2, Item 3, Item 4 and Item 5 to know where 1A ends
    pattern = re.compile(r'(((>|\n)(|\s+)i(|(<[^>]+>)+)t(|(<[^>]+>)+)e(|(<[^>]+>)+)m)'
                           '(\s+|&#160;|&nbsp;| </b><b>|(<[^>]+>)+|&#xA0;|&#xa0;)(|&nbsp;|&#xa0;|&#xA0;|&#160;|(<[^>]+>)+|\s+)'
                           '(|(<[^>]+>)+)(1(|(<[^>]+>)+)(|\s+)(A|a)|1(|(<[^>]+>)+)(|\s+)(B|b)|2|3|4|5))')
    
    # Use finditer to math the regex
    matches = pattern.finditer(document)

    # Create a dataframe with start and end of each instance of Item 1A and Item 1B found
    df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])
    try:
        df.columns = ['item', 'start', 'end']
    except:
        return format_('ITEM 1A NOT FOUND',ticker)
    #df['item'] = df.item.str.lower()
    
    # Get rid of unnesesary characters from the dataframe
    df.replace('&#160;',' ',regex=True,inplace=True)
    df.replace('&nbsp;',' ',regex=True,inplace=True)
    df.replace('&#xa0;',' ',regex=True,inplace=True)
    #df.replace('</b><b>','',regex=True,inplace=True)
    #df.replace('>tem','item',regex = True,inplace=True)
    #df.replace('>em','item',regex = True,inplace=True)
    df.replace('<[^>]+>','',regex = True,inplace = True)
    df.replace(' ','',regex=True,inplace=True)
    df.replace('\.','',regex=True,inplace=True)
    df.replace('>','',regex=True,inplace=True)
    df.replace('\n','',regex=True,inplace=True)
    
    #if no instance of item 1a found
    if(not len(df[df.item == 'item1a'])):
        return format_('ITEM 1A NOT FOUND',ticker)
    
    #if several instances of item 1a are found, remove the ones that are after all the other items
    #since won't know the end of item 1a in that case
    if(len(df[df.item == 'item1a']) >= 2):
        while((df[df.start == df.start.max()].item == 'item1a').values and (len(df[df.item == 'item1a']) >= 2)):
            df = df[df.start != df.start.max()]

    # Drop duplicates keeping the last 
    df.sort_values('start', ascending = True, inplace = True)
    df_unique = df.drop_duplicates(subset=['item'], keep='last').copy()
    
    #find the item that is right after item 1a
    next_item = find_next_item(df_unique)
         
    if(next_item == 'NOT FOUND'):
        return format_('ITEM 1A NOT FOUND', ticker)
    
    # Set item as the dataframe index
    df_unique.set_index('item', inplace=True)
    
    item_1a_raw, correct_item_found = extract_item_1a(df_unique, document, next_item)
    
    #if the the text identified does not seem to be correct
    #we can test and see whether the previous instance that was found is the right one
    if not correct_item_found:
        if (len(df[df.item == 'item1a']) > 1):
            #delete the bounds that were previously considered
            df = pd.merge(df,df_unique, how = 'outer', indicator=True)
            df = df[df._merge != 'both'].drop('_merge', axis=1).reset_index(drop=True)
            
            #if item 1a is now the last element, add back the previously considered bounds for the other items
            if((df[df.start == df.start.max()].item == 'item1a').values):
                df_unique = df_unique.reset_index()
                df_unique = df_unique[df_unique.item != 'item1a']
                df = pd.merge(df, df_unique, how = 'outer')

            #re-exctract item 1a using the new bounds
            df_unique = df.drop_duplicates(subset=['item'], keep='last').copy()
            #find the item that is right after item 1a
            next_item = find_next_item(df_unique)

            if(next_item == 'NOT FOUND'):
                return format_('ITEM 1A NOT FOUND', ticker)

            # Set item as the dataframe index
            df_unique.set_index('item', inplace=True)

            item_1a_raw, correct_item_found = extract_item_1a(df_unique, document, next_item)

        if not correct_item_found:
            return format_('ITEM 1A NOT FOUND', ticker)
    
    #remove leading and trailing whitespaces
    item_1a_raw = item_1a_raw.strip()
    
    time.sleep(0.6)
    return format_(item_1a_raw, ticker)

#function that takes as input a dataframe of 
def find_next_item(df):
    #if there is no item 1b, risk factors ends where item 2 starts,
    #if there is also no item 2, risk factors ends where item 3 starts,...
    items = ['item1b', 'item2', 'item3', 'item4', 'item5', 'NOT FOUND']
    next_item = items[0]
    for item in items[1:]:
        #if the item that is currently considered to be the item after item 1a
        #does not exist, consider the item after that one
        if not len(df[df.item == next_item]):
            next_item = item
        #if the item that is currently considered to be the item after item 1a
        #exists but before item 1a, it is probably only mentionned in the table of contents,
        #hence consider the item after that one
        elif df[df.item == next_item].start.values < df[df.item == 'item1a'].end.values :
            next_item = item
        #if the previous cases are not verified, next_item is the correct one
        else:
            break
    
    return next_item


#function that extracts item 1a from the 10-K statement
#and checks whether the extracted text is the correct one
def extract_item_1a(df_unique, document, next_item):
    # Get Item 1a
    item_1a_raw = document[int(df_unique['start'].loc['item1a']):int(df_unique['start'].loc[next_item])]
    
    #safety check:
    #if the identified text is short it is probably wrong
    #unless it is just a stament explaining that the company is legally not required
    #to disclose risk factors as it is a small company
    correct_item_found = True
    item_1a = BeautifulSoup(item_1a_raw, 'lxml').get_text()
    if(len(item_1a) < 500):
        correct_item_found = False
        small_company_words = ['smaller', 'reporting', 'company']
        small_company = all([word in item_1a.lower() for word in small_company_words])
        small_company_words_v2 = ['not', 'applicable']
        small_company_v2 = all([word in item_1a.lower() for word in small_company_words_v2])
        if small_company or small_company_v2:
            correct_item_found = True
    return item_1a_raw, correct_item_found


#function that adds delimiters before and after the text
def format_(text, ticker):
    start_pattern = "====={}=====\n".format(ticker)
    end_pattern = "\n====={}=====\n\n\n".format(ticker)
    text = start_pattern + text + end_pattern
    return text
    
    
#function that cleans and tokenizes a text (single string)
def clean(text, remove_html = True, remove_numbers = True, lower_case = True,
          remove_punctuation = True, remove_sep_letters = True, remove_paragraph_nums = True,
          remove_stopwords = True, tokenize_words = True, tokenize_sentences = False, stemming = False,
          remove_common = False):
    
    #check if the text was found or not
    if text == 'ITEM 1A NOT FOUND':
        return False
    
    #remove html stuff
    if(remove_html):
        text = BeautifulSoup(text, 'lxml').get_text()
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'\xa0;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        #text = re.sub('\\x', ' ', text)
    
    #remove numbers
    if(remove_numbers):
        text = re.sub(r'\d+', '', text)
    
    #lower case all letters
    if(lower_case):
        text = text.lower()
    
    #remove punctuation
    if(remove_punctuation):
        if(not tokenize_sentences):
            punctuation = '!@#$%^&*()_−-+={}[]:;"\'|<>,.?/~`—’•“”✓●·–ÿ■​✔®☒☐§¨ø≥'
        else:
            punctuation = '@#$%^&*()_−-+={}[]"\'|<>/~`—’•“”✓●·–ÿ■​✔®☒☐§¨ø≥'
        for marker in punctuation:
            text = text.replace(marker, " ")
       
    #tokenize text into words
    if(tokenize_words):
        text = word_tokenize(text)
    
        #remove stop words
        if(remove_stopwords):
            stop_words = stopwords.words('english')
            text = [word for word in text if word not in stop_words]

        #remove paragraph numbering
        if(remove_paragraph_nums):
            to_remove = ['ii', 'iii', 'iv','vi','vii','viii','ix']
            text = [word for word in text if word not in to_remove]

        #remove letters that get separated
        if(remove_sep_letters):
            text = [word for word in text if len(word) != 1]
        
        #remove non-alphabet characters from the words
        alphabet = re.compile('[^a-z]')
        text = [alphabet.sub('',word) for word in text]
        
        #remove common/frequent words
        if(remove_common):
            common_words = top_n_list('en', 100)
            text = [word for word in text if word not in common_words]
        
        #apply stemming
        if(stemming):
            text = stem(text)
    
    #tokenize text into sentences
    elif(tokenize_sentences):
        text = re.sub('\n', ' ', text)
        text = re.sub('\s+', ' ', text)
        
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        text = tokenizer.tokenize(text)

    
    return text


#function that saves the cleaned risk disclosure of a firm for a given year
def save_text(year, path, text, item1a = True):
    if item1a:
        file_name = "{}_item_1a.txt".format(year)
    else:
        file_name = "{}_cyber_sentences.txt".format(year)
    with open(path+file_name, "a") as f:
        f.write(text)
    return


#function that takes as input a raw risk factor statement and saves the cybersecurity related sentences
@dask.delayed()
def find_cybersecurity_sentences(raw_1a, cyber_keywords):
    #extract the company's ticker
    ticker = re.findall('=====([^=]+)=====', raw_1a)[0]
    raw_1a = re.sub('=====([^=]+)=====', '', raw_1a)
    clean_1a = clean(raw_1a, tokenize_words = False, tokenize_sentences = True, remove_numbers = False)
    
    #find sentences with direct description of cybersecurity risks
    direct_sentences = []
    idx_direct = []
    direct_keywords = cyber_keywords[cyber_keywords.description == 'direct'].Keyword.unique()
    for sentence, i in zip(clean_1a, range(len(clean_1a))):
        #apply stemming to the words from the risk statements
        words = word_tokenize(sentence)
        words = stem(words)
        
        if cyber_related(words, cyber_keywords[cyber_keywords.description == 'direct'], direct_keywords):
            direct_sentences.append(sentence)
            idx_direct.append(i)
    
    indirect_sentences = []
    if(len(direct_sentences)):
        #find the indexes of the 10 sentences following each sentence with a direct description
        idx_indirect = []
        for idx in idx_direct:
            idx_indirect.append(np.arange(idx+1,idx+11))
        #make sure there are no overlaps and ignore the indexes corresponding to sentences with a direct description
        idx_indirect = np.unique(list(np.concatenate(idx_indirect).flat))
        idx_indirect = list(set(idx_indirect).difference(idx_direct))
        idx_indirect = [i for i in idx_indirect if i < len(clean_1a)]

        #find sentences with indirect description of cybersecurity risks
        indirect_keywords = cyber_keywords[cyber_keywords.description == 'indirect'].Keyword.unique()
        for sentence in [clean_1a[idx] for idx in idx_indirect]:
            #apply stemming to the words from the risk statements
            words = word_tokenize(sentence)
            words = stem(words)
            if cyber_related(words, cyber_keywords[cyber_keywords.description == 'indirect'], indirect_keywords):
                indirect_sentences.append(sentence)
        
        to_save = ' '.join(direct_sentences) +"\n\n" +' '.join(indirect_sentences)
        #remove leading and trailing whitespaces
        to_save = to_save.strip()
        
        return format_(to_save, ticker)
    
    return format_('NO CYBER RELATED SENTENCES', ticker)
        
    
#function that determines whether a sentence is cybersecurity related
def cyber_related(words, cyber_keywords, keywords):
    for word in keywords:
        #Keyword could be a combination of words (ex:information systems)
        if np.isin(stem(word_tokenize(word)), words).all():
            relevant_words = cyber_keywords[cyber_keywords.Keyword == word].Relevant_hit.dropna().values
            relevant_words = [word_tokenize(rw) for rw in relevant_words]
            if len(relevant_words):
                #apply stemming to the keywords
                relevant_words = [stem(rw) for rw in relevant_words]
                #differentiate between relevant keys that are only one words and the ones that are expressions
                relevant_words_1word = np.squeeze([rw for rw in relevant_words if len(rw) < 2])
                relevant_words_expression = [rw for rw in relevant_words if len(rw) >= 2]
            else:
                #if no relevant words needed, set relevant words to the words of the statement
                #so any match of the keywords is relevant
                relevant_words_1word = words
                relevant_words_expression = []

            irrelevant_words = cyber_keywords[cyber_keywords.Keyword == word].Irrelevant_hit.dropna().values
            irrelevant_words = [word_tokenize(irw) for irw in irrelevant_words]
            if len(irrelevant_words):
                #apply stemming to the keywords
                irrelevant_words = [stem(irw) for irw in irrelevant_words]
                #differentiate between relevant keys that are only one words and the ones that are expressions
                irrelevant_words_1word = np.squeeze([irw for irw in irrelevant_words if len(irw) < 2])
                irrelevant_words_expression = [irw for irw in irrelevant_words if len(irw) >= 2]
            else:
                #if no irrelevant words needed, set irrelevant words to an empty string
                #so no match of the keywords is irrelevant
                irrelevant_words_1word = ['']
                irrelevant_words_expression = []

            #check if sentence is relevant or not
            relevant_1word = np.isin(relevant_words_1word, words).any() #True = relevant
            irrelevant_1word = np.isin(irrelevant_words_1word, words).any() #True = irrelevant
            if len(relevant_words_expression):
                relevant_expression = any([np.isin(rwe, words).all() for rwe in relevant_words_expression])
            else:
                relevant_expression = False
            if len(irrelevant_words_expression):
                irrelevant_expression = any([np.isin(iwe, words).all() for iwe in irrelevant_words_expression])
            else:
                irrelevant_expression = False

            relevant = relevant_1word or relevant_expression #True if sentence is relevant
            irrelevant = irrelevant_1word or irrelevant_expression #True if sentence is irrelevant

            if(relevant and not irrelevant):
                return True
    return False
    

def get_alphas_old(low_tercile_returns, middle_tercile_returns, high_tercile_returns, FF5):
    table = pd.DataFrame(np.ones([4,4]), index = ['Excess return', 'CAPM alpha', 'FFC alpha','FF5 alpha'],
                    columns = ['T1 (low)', 'T2', 'T3 (high)', 'T3-T1'])
    pval_table = table.copy()*np.nan
    idx = low_tercile_returns.index
    FFC_factors = ['Mkt-RF','HML','SMB','UMD']
    FF5_factors = ['Mkt-RF','HML','SMB','RMW', 'CMA']
    
    #excess returns
    low_er = low_tercile_returns.sub(FF5.loc[idx, 'RF'], axis = 0)
    middle_er = middle_tercile_returns.sub(FF5.loc[idx, 'RF'], axis = 0)
    high_er = high_tercile_returns.sub(FF5.loc[idx, 'RF'], axis = 0)
    long_short_er = high_er.sub(low_er.values, axis = 0)
    table.iloc[0,0] = low_er.mean()
    table.iloc[0,1] = middle_er.mean()
    table.iloc[0,2] = high_er.mean()
    table.iloc[0,3] = long_short_er.mean()
    pval_table.iloc[0,0] = sm.OLS(low_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const
    pval_table.iloc[0,1] = sm.OLS(middle_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const
    pval_table.iloc[0,2] = sm.OLS(high_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const
    pval_table.iloc[0,3] = sm.OLS(long_short_er,np.ones(len(idx))).fit(cov_type='HAC',cov_kwds={'maxlags':12}).pvalues.const

    #CAPM alpha
    low_capm = sm.OLS(low_er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    middle_capm = sm.OLS(middle_er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    high_capm = sm.OLS(high_er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    long_short_capm = sm.OLS(long_short_er,sm.add_constant(FF5.loc[idx, 'Mkt-RF'])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[1,0] = low_capm.params.const
    table.iloc[1,1] = middle_capm.params.const
    table.iloc[1,2] = high_capm.params.const
    table.iloc[1,3] = long_short_capm.params.const
    pval_table.iloc[1,0] = low_capm.pvalues.const
    pval_table.iloc[1,1] = middle_capm.pvalues.const
    pval_table.iloc[1,2] = high_capm.pvalues.const
    pval_table.iloc[1,3] = long_short_capm.pvalues.const

    #FFC alpha
    low_ffc = sm.OLS(low_er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    middle_ffc = sm.OLS(middle_er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    high_ffc = sm.OLS(high_er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    long_short_ffc = sm.OLS(long_short_er,sm.add_constant(FF5.loc[idx, FFC_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[2,0] = low_ffc.params.const
    table.iloc[2,1] = middle_ffc.params.const
    table.iloc[2,2] = high_ffc.params.const
    table.iloc[2,3] = long_short_ffc.params.const
    pval_table.iloc[2,0] = low_ffc.pvalues.const
    pval_table.iloc[2,1] = middle_ffc.pvalues.const
    pval_table.iloc[2,2] = high_ffc.pvalues.const
    pval_table.iloc[2,3] = long_short_ffc.pvalues.const

    #FF5 alpha
    low_ff5 = sm.OLS(low_er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    middle_ff5 = sm.OLS(middle_er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    high_ff5 = sm.OLS(high_er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    long_short_ff5 = sm.OLS(long_short_er,sm.add_constant(FF5.loc[idx, FF5_factors])).fit(cov_type='HAC',cov_kwds={'maxlags':12})
    table.iloc[3,0] = low_ff5.params.const
    table.iloc[3,1] = middle_ff5.params.const
    table.iloc[3,2] = high_ff5.params.const
    table.iloc[3,3] = long_short_ff5.params.const
    pval_table.iloc[3,0] = low_ff5.pvalues.const
    pval_table.iloc[3,1] = middle_ff5.pvalues.const
    pval_table.iloc[3,2] = high_ff5.pvalues.const
    pval_table.iloc[3,3] = long_short_ff5.pvalues.const

    return table, pval_table
