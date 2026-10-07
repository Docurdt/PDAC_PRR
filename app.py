"""Local interface for prepared, numeric multimodal feature uploads."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import streamlit as st

from pdac_prr.inference import Predictor

st.set_page_config(page_title='PDAC PRR', layout='centered')
st.title('PDAC PRR')
st.caption('Research inference from the retained ESC and PRR checkpoints.')
family = st.selectbox('Model', ['prrpath', 'prrpath_text', 'prrfull', 'esc'])
st.write('Upload a numeric NPZ file containing the features required by the selected model.')
st.write('PRRpath: ls_features (256). PRRpath+text adds text_features (768). PRRfull adds clin_features (4).')
uploaded = st.file_uploader('Feature file', type=['npz'])
if st.button('Predict', disabled=uploaded is None):
    try:
        with np.load(BytesIO(uploaded.getvalue()), allow_pickle=False) as data:
            if family == 'esc':
                bags = {k: {'features': data[k]} for k in ['gigapth','hoptimus0','ctranspath','uni_v2']}
            else:
                bags = {k: data[k] for k in data.files}
            with st.spinner('Running model ensemble…'):
                result = Predictor(Path(__file__).parent).predict(family, bags)
        st.dataframe({'score': result['scores']}, hide_index=True)
        st.caption('PRR scores rank prognosis; they are not calibrated survival probabilities.')
        st.download_button('Download results', json.dumps(result, indent=2),
                           file_name='pdac_prr_prediction.json', mime='application/json')
    except Exception as exc:
        st.error(str(exc))
