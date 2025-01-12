from functions import *
def t_Garch(filepath, nom, dist):
    
    AKD = pd.read_excel(filepath)
    AKD.head()
    AKD =data_clean(AKD)
    AKD.head()
    plotdata(AKD, nom)
    akd = toserie(AKD)
    adf(akd)
    archtest(akd)
    rend = rendement(AKD)
    adf(rend)
    archtest(rend)
    plotrendement(rend, nom)
    model = bestmodel(rend, 10, 10, dist)
    print(model.summary())
    dof(model, dist)
    validation(model, nom)
    ks_test_skewt_2(model)
    residuplot(model, nom)
    ajustementplot(rend, model, nom, dist)
    return model

    
AKD = t_Garch('AKDITAL.xlsx', 'AKDITAL', 'skewt')

ATW = t_Garch('ATW.xlsx', "ATTIJARI", 'skewt')

BCP = t_Garch('BCP.xlsx', 'Banque Populaire', 'skewt')      

resid_akd = AKD.std_resid
resid_atw = ATW.std_resid
resid_bcp = BCP.std_resid
print(AKD.params['eta'], AKD.params['lambda'])
print(ATW.params['eta'], ATW.params['lambda'])
print(BCP.params['eta'], BCP.params['lambda'])



data = pd.DataFrame({
    "Akdital" : resid_akd,
    "ATTIJARI" : resid_atw,
    "BCP" : resid_bcp
})
data
data.to_csv('Resiuds.csv')
