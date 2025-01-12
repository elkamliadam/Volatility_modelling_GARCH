
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from scipy.stats import normaltest, jarque_bera, shapiro



def data_clean(data):
    data['Séance'] = pd.to_datetime(data['Séance'], format = "%d/%m/%Y")
    data = data[['Séance', 'Cours ajusté']]
    data.columns = ['Date', 'Prix']
    data.dropna(subset = ['Date', 'Prix'], inplace=True)
    return data

def plotdata(data, nom):
    plt.plot(data.Date, data.Prix)
    plt.title(f"Données Historique de l'action {nom}",fontweight = 'bold')
    plt.xlabel("Date")
    plt.ylabel("Prix ajusté")
    plt.show()
    
    


def toserie(data):
    data.set_index('Date', inplace=True)
    serie = data['Prix']
    return serie

def adf(serie):
    adf = adfuller(serie)
    print(f'\033[1mTest de Dicky-Fuller : P-value = {adf[1]}\033[0m')
    if adf[1] < 0.05:
        print("\nLa série est stationnaire")
    else :
        print("\nLa série n'est pas stationnaire")
        

def archtest(serie):
    arch_test = het_arch(serie)
    print('---------------------------------------------------------')
    print('\n\033[1mTest d\'hétéroscédasticité conditionnelle\033[0m')
    print(f'\np-value = {arch_test[1]}')



def rendement(data):
    data['return'] = data['Prix'].pct_change()
    data = data.dropna(subset=['return'])
    serie = data['return']
    serie.head()
    return serie


def plotrendement(serie, nom):
    plt.plot(serie)
    plt.title(f"Série des rendements de l'action {nom}", fontweight = 'bold')
    plt.xlabel("Date")
    plt.ylabel("Rendement")
    plt.grid()
    plt.tight_layout()
    plt.show()


def bestmodel(serie, plim, qlim, dist):
    serie = serie*100
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in range(1, plim):
        for q in range( 1, qlim):
            model = arch_model(serie, vol = 'Garch', p=p, q=q, dist = dist)
            model_fit = model.fit(disp = 'off')
            if model_fit.aic < best_aic :
                best_aic = model_fit.aic
                best_order = (p,q)
                best_model = model_fit
    print(f'Meilleur modèle de GARCH trouvé : p ={best_order[0]}, q = {best_order[1]} avec distribution {dist}')
    return best_model


def dof(model, dist):
    if dist == "StudentsT":
        dof = model.params.get('nu')
    else : 
        dof = model.params.get('eta')
    print(f"Degré de liberté : {dof}")
    if dof <= 3:
        print('Cette série à une forte queue lourde')
    elif dof <= 10 and dof > 3:
        print('Cette série à un queue lourde')
    elif dof >= 30 : 
        print('Cette série ne présent pas de queue lourde et elle est plutôt noramle')
    return dof

def validation(model, nom):
    resid = model.std_resid
    ljung = acorr_ljungbox(resid, lags = 1)
    print('--------------------------------------------')
    print(f"\n\033[1mTest de Ljung-Box de la série de {nom} :statistique et  p-value:\n\033[0m ")
    print(ljung)
    jb_stat, jb_p_value = jarque_bera(resid)
    print('--------------------------------------------')
    print(f"\n\033[1mTest de normalité des résidus de la série de {nom} (Jarque-Bera):\n\033[0m")
    print(f"p-value: {jb_p_value}")
    print(f"statistique de jarque Bera{jb_stat}")
    stat, p_v = shapiro(resid)
    print("--------------------------------------------")
    print(f"\n\033[1mTest de normalité des résidus de la série de {nom} (Shapiro-Wilk):\n\033[0m")
    print(f'P-alue : {p_v}')
    print(f'Statistique de Shapiro-wilk : {stat}')


import scipy.stats as stats
def resid_stud(model, nom):
    
    dof = model.params.get('nu')
    resid  = model.resid/ model.conditional_volatility
    stats.probplot(resid, dist = "t", sparams=(dof,), plot = plt)
    plt.title(f'Q-Q plot des Résidus de {nom} vs t-Student')
    plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ks_test_skewt_2(model):
    residuals = model.std_resid
    dof = model.params.get('eta')
    skew = model.params.get('lambda')
    # Définir la fonction de densité pour la distribution skewed-t
    def skewt_pdf(x, nu, lambda_):
        # Calculer la densité pour la skewed-t
        return stats.t.pdf(x, nu) + lambda_ * (1 - stats.t.cdf(x, nu)) * stats.t.pdf(x, nu)
    
    # Estimation de la densité empirique des résidus
    # Utilisation de l'histogramme pour obtenir la densité empirique
    density_empirical, bins, _ = plt.hist(residuals, bins=30, density=True, alpha=0.6, color='blue', label="Densité empirique")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calcul de la densité théorique pour la skewed-t
    density_theoretical = skewt_pdf(bin_centers, dof, skew)
    
    # Affichage de la comparaison entre les densités
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, density_empirical, label="Densité empirique", color='blue', linestyle='-', marker='o', markersize=5)
    plt.plot(bin_centers, density_theoretical, label="Densité théorique skewed-t", color='red', linestyle='--')
    plt.title("Comparaison des Densités Empirique et Théorique (Skewed-t Distribution)")
    plt.xlabel("Résidus")
    plt.ylabel("Densité")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
    
    # Effectuer le test Kolmogorov-Smirnov (comparaison entre la densité empirique et théorique)
    ks_statistic = np.max(np.abs(density_empirical - density_theoretical))
    p_value = 1 - ks_statistic  # Simplification, le p-value dépend du KS statistic
    
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {p_value}")
    
    # Interprétation basée sur le p-value
    if p_value > 0.05:
        print("Les résidus ne dévient pas de manière significative de la distribution skewed-t.")
    else:
        print("Les résidus dévient de manière significative de la distribution skewed-t.")



    

def residuplot(model, nom):
    std_residuals = model.std_resid
    plt.figure(figsize=(10,6))
    plt.hist(std_residuals, bins=30, density=True, alpha=0.6, color='g')
    plt.title(f'Histogramme des résidus standardisés de la série {nom}', fontweight = 'bold')
    plt.show()


def ajustementplot(serie,model, nom, dist):
    serie= serie*100
    plt.figure(figsize=(10, 6))
    plt.plot(serie, label=f'Série des rendements de {nom}')
    plt.plot(model.conditional_volatility, label='Volatilité conditionnelle', color='red')
    plt.legend()
    plt.title(f'Volatilité conditionnelle estimée par le modèle GARCH pour le rendement de {nom} et une distribution de {dist}')
    plt.xlabel('Date')
    plt.ylabel('Rendement')
    plt.show()





