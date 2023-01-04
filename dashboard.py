from dash import Dash, dash_table, html, dcc
import pandas as pd

global df
df = pd.read_csv('recurso/Pacients.csv')
df = df.iloc[:,1:41]
df = df.loc[:, ['Name_Doctor', 'Id_Doctor','Name_Patient','Id_Patient','Result_1','Result_2','Result_3','Result_4','Result_5','Data']]
app = Dash(__name__)

def finder():
    name = dcc.Input('Name_Patient', type='text',placeholder='')
    return df.loc[(df['Name_Doctor']) == name]


app.layout = html.Div([
    html.H1('Pacients'),
    html.Div([dcc.Input(name='Name_Patient', type='text',placeholder='Name_pacient')]),
    html.Button(type="button", name='findding',title='finder'),
                 dash_table.DataTable(df.to_dict('records'),
                                  [{"name": i, "id": i} for i in df.columns],
                                  page_size=20,
                                  style_table={'height': '500px','overflowY': 'auto'},
                                  style_header={
                                      'backgroundColor': 'rgb(30, 30, 30)',
                                      'color': 'white'},
                                  style_data={
                                      'backgroundColor': 'rgb(50, 50, 50)',
                                      'color': 'white'
                                  })])


if __name__ == '__main__':
    app.run_server(debug=True)