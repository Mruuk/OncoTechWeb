from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from keras.models import model_from_json
import datetime as dt

arquivo = open('recurso/classificador_breast.json', 'rb')
estrutura_rede = arquivo.read()
arquivo.close()
classificador = model_from_json(estrutura_rede)
classificador.load_weights('recurso/classificador_breast.h5')
previsores = pd.read_csv('recurso/entradas_breast.csv')
classe = pd.read_csv('recurso/saidas_breast.csv')
classificador.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)

app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/previsao", methods=['POST'])
def previsao():

    global registro
    registro = np.array(
        [['nome_medico', 'id_m001', 'nome_paciente', 'id_p001', 15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
          0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
          0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
          0.84, 158, 0.363]])

    global name_pacient
    global name_doctor
    global id_pacient
    global id_doctor

    data = request.form.get
    registro[0][0] = data('name_doctor')
    name_doctor = registro[0][0]
    registro[0][1] = data('id_doctor')
    id_doctor = registro[0][1]
    registro[0][2] = data('name_pacient')
    name_pacient = registro[0][2]
    registro[0][3] = data('id_pacient')
    id_pacient = registro[0][3]
    registro[0][4] = float(data('radius_mean'))
    registro[0][5] = float(data('texture_mean'))
    registro[0][6] = float(data('perimeter_mean'))
    registro[0][7] = float(data('area_mean'))
    registro[0][8] = float(data('smoothness_mean'))
    registro[0][9] = float(data('compactness_mean'))
    registro[0][10] = float(data('concavity_mean'))
    registro[0][11] = float(data('concave_points_mean'))
    registro[0][12] = float(data('symmetry_mean'))
    registro[0][13] = float(data('fractal_dimension_mean'))
    registro[0][14] = float(data('radius_se'))
    registro[0][15] = float(data('texture_se'))
    registro[0][16] = float(data('perimeter_se'))
    registro[0][17] = float(data('area_se'))
    registro[0][18] = float(data('smoothness_se'))
    registro[0][19] = float(data('compactness_se'))
    registro[0][20] = float(data('concavity_se'))
    registro[0][21] = float(data('concave_points_se'))
    registro[0][22] = float(data('symmetry_se'))
    registro[0][23] = float(data('fractal_dimension_se'))
    registro[0][24] = float(data('radius_worst'))
    registro[0][25] = float(data('texture_worst'))
    registro[0][26] = float(data('perimeter_worst'))
    registro[0][27] = float(data('area_worst'))
    registro[0][28] = float(data('smoothness_worst'))
    registro[0][29] = float(data('compactness_worst'))
    registro[0][30] = float(data('concavity_worst'))
    registro[0][31] = float(data('concave_points_worst'))
    registro[0][32] = float(data('symmetry_worst'))
    registro[0][33] = float(data('fractal_dimension_worst'))

    data_log = dt.datetime.now()
    data_log = data_log.strftime("%d/%m/%y %H:%M")

    print(registro)
    breast = registro[0][4:34]
    breast = list(map(float, breast))
    breast = np.array([breast])

    previsao = classificador.predict(breast)
    previsao = (previsao > 0.9)
    amostra = data('amostras')
    if amostra == '1':
        global previsao_1
        previsao_1= previsao
        return render_template("homepage.html", previsao_1=previsao, name_doctor=name_doctor, id_doctor=id_doctor, name_pacient=name_pacient, id_pacient=id_pacient)
    if amostra == '2':
        global previsao_2
        previsao_2= previsao
        return render_template("homepage.html", previsao_1=previsao_1, previsao_2=previsao, name_doctor=name_doctor, id_doctor=id_doctor, name_pacient=name_pacient, id_pacient=id_pacient)
    if amostra == '3':
        global previsao_3
        previsao_3= previsao
        return render_template("homepage.html", previsao_1=previsao_1, previsao_2=previsao_2, previsao_3=previsao, name_doctor=name_doctor, id_doctor=id_doctor, name_pacient=name_pacient, id_pacient=id_pacient)
    if amostra == '4':
        global previsao_4
        previsao_4= previsao
        return render_template("homepage.html", previsao_1=previsao_1, previsao_2=previsao_2, previsao_3=previsao_3, previsao_4=previsao, name_doctor=name_doctor, id_doctor=id_doctor, name_pacient=name_pacient, id_pacient=id_pacient)
    if amostra == '5':
        global previsao_5
        previsao_5= previsao
        global data_exame
        data_exame = pd.Timestamp.now()
        data_exame = data_exame.strftime("%m/%d/%y %H:%M")
        print(f'paciente:{name_pacient}, medico: {name_doctor}, amostra 1: {previsao_1}, amostra 2: {previsao_2}, amostra 3: {previsao_3}, amostra 4: {previsao_4}, amostra 5: {previsao_5}')
        return render_template("homepage.html", previsao_1=previsao_1, previsao_2=previsao_2, previsao_3=previsao_3, previsao_4=previsao_4, previsao_5=previsao, name_doctor=name_doctor, id_doctor=id_doctor, name_pacient=name_pacient, id_pacient=id_pacient, data_exame=data_exame)



@app.route("/consulta")
def consulta():
    base = pd.DataFrame(registro)
    base.rename(columns={0: 'Name_Doctor', 1: 'Id_Doctor', 2: 'Name_Patient', 3: 'Id_Patient'}, inplace=True)
    for i in range(registro.shape[1] - 4):
        base.rename(columns={i + 4: previsores.columns[i]}, inplace=True)
    base.loc[:, 'Result_1'] = previsao_1
    base.loc[:, 'Result_2'] = previsao_2
    base.loc[:, 'Result_3'] = previsao_3
    base.loc[:, 'Result_4'] = previsao_4
    base.loc[:, 'Result_5'] = previsao_5
    base.loc[:, 'Data'] = data_exame
    base.to_csv("recurso/Pacients.csv", mode='a', header=False)
    print(base)
    return render_template("consulta.html", previsao_1=previsao_1, previsao_2=previsao_2, previsao_3=previsao_3, previsao_4=previsao_4, previsao_5=previsao_5, name_doctor=name_doctor, id_doctor=id_doctor, name_pacient=name_pacient, id_pacient=id_pacient, data_exame=data_exame)

@app.route("/pacients")
def pacient():
    return render_template("pacient.html", name_pacient=name_pacient)

if __name__ == "__main__":
    app.run(debug=True)

