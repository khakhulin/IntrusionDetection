import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Data for processing ########################################################
corrected = "./data/corrected"
kddcup_data_corrected = "./data/kddcup.data.corrected"

kddcup_data_10_percent_corrected = "./data/kddcup.data"
kddcup_testdata_10_percent = "./data/kddcup.testdata.unlabeled_10_percent"

correspondence = "./data/training_attack_types.txt"
header_files = "./data/headers"
# Auxiliary functions ########################################################
# This function turns categorical features into labels(integer numbers)
def labels_map(label):
    """
     0 - normal
     1 - probe -  surveillance and other probing, e.g., port scanning.
     2 - dos - denial-of-service, e.g. syn flood;
     3 - u2r - unauthorized access to local superuser (root) privileges, e.g., various __buffer overflow__ attacks;
     4 - R2L - unauthorized access from a remote machine, e.g. guessing password;
    """
    label = str(label).split('.')[0]
    if label == 'normal':
        return 0
    if label in ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']: 
        return 1
    if label in ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']:
        return 2
    if label in ['buffer_overflow', 'httptunnel', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']: 
        return 3
    if label in ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack',
                 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop']: 
        return 4
# Function transforms continuous data into (0;1) interval
def do_normalization(x_train):
    cat_features = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login']
    cols=list(x_train.axes[1].values)
    #categorial features have integer type(after labelling) and must be normalized
    cont_features=cols#list(set(cols).difference(set(cat_features)))
    scaler = MinMaxScaler()
    x_train.loc[:,cont_features]=scaler.fit_transform(x_train.loc[:,cont_features])
    return x_train
##############################################################################
def preprocess_data(train_data_path, test_data_path, header_file, correspondence, save_to_file=False, do_normalization=False,
                    x_train_path="./data/train_data", y_train_path="./data/train_labels",
                    x_test_path="./data/test_data", y_test_path="./data/test_labels"):
    dct = dict()

    with open(correspondence, "r") as f:
        for line in f:
            line = line.split()
            if line != []:
                dct[line[0]] = line[1][:-1]
                
    dct["normal"] = "normal"

    with open(header_file, 'r') as f:
        header = f.readline().strip().split(',')

    df = pd.read_csv(train_data_path, header=None)
    df.columns = header

    test_data = pd.read_csv(test_data_path, header=None)
    test_data.columns = header[:-1]

    df['classes'] = df.classes.apply(labels_map)
    test_data.classes = test_data.apply(labels_map)

    rows_count = df.shape[0]
    df.groupby('classes').size() * 100 / rows_count

    X = df[df.columns[:-1]]

    categorial_features = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login']
    x_categor = X.loc[:,categorial_features]

    # In connection with a large number of duplicate data, we delete them.
    # Remove duplicate rows

    df = df.drop_duplicates()
    df.num_outbound_cmds.unique()

    # This feature has the same value in all samples.

    df.drop('num_outbound_cmds', axis=1, inplace=True)

    cols=list(df.axes[1].values)
    x_numerical=df.loc[:,list(set(cols).difference(set(categorial_features)))]

    # We should remove unneccessary information
    corr_column_names = ['dst_host_srv_serror_rate', 'srv_serror_rate', 'dst_host_serror_rate']
    df.drop(corr_column_names, axis=1, inplace=True)
    # Labeling categorial features
    le = LabelEncoder()
    need_labeling = ['protocol_type', 'service','flag']

    for i in need_labeling:
        le.fit(df[i])
        df[i] = le.transform(df[i])

    # Feature selection using random forest
    rf = RandomForestClassifier(max_depth=25, random_state=42)

    X = df[df.columns[:-1]]
    y = df['classes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

    if do_normalization:
        X_train = do_normalization(X_train)
        X_test = do_normalization(X_test)

    rf.fit(X_train, y_train)
    # Save data
    if save_to_file:
        X_train.to_csv(x_train_path)
        y_train.to_csv(y_train_path)
        X_test.to_csv(x_test_path)
        y_test.to_csv(y_test_path)
    else:
        return X_train, y_train, X_test, y_test, [(df.columns.get_loc(name) - 1) for name in categorial_features]