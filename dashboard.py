from dash import Dash, dash_table, html, dcc
import pandas as pd

df = pd.read_csv('recurso/Pacients.csv')
df = df.iloc[:,1:41]
df = df.loc[:, ['Name_Doctor', 'Id_Doctor','Name_Patient','Id_Patient','Result_1','Result_2','Result_3','Result_4','Result_5','Data']]
app = Dash(__name__)
html.Div([html.H1('Hello Dash')])
dcc.Markdown("teste")
app.layout = dash_table.DataTable(df.to_dict('records'),
                                  [{"name": i, "id": i} for i in df.columns],
                                  page_size=20,
                                  style_table={'height': '500px','overflowY': 'auto'},
                                  style_header={
                                      'backgroundColor': 'rgb(30, 30, 30)',
                                      'color': 'white'},
                                  style_data={
                                      'backgroundColor': 'rgb(50, 50, 50)',
                                      'color': 'white'
                                  })


if __name__ == '__main__':
    app.run_server(debug=True)