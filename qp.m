function qp2(inFile, outFile)
    load('data/qp_data.mat');
    %% Contient:
    % G : Mat(y[i] * y[j] * k(x[i], x[j])) 
    %     -> matrice (n,n)
    % yApp : labels des x[i] sur lesquels repose G 
    %        -> vecteur ligne ie (1, n)
    % yTest : labels des données de tests
    %         -> vecteur ligne ie (1, m)
    % K : Mat(k(x, x[i])) pour x dans les données d'apprentissage
    %     et de test
    %     -> matrice (n+m, n)

    %% Utilisation de monqp:
    % function [xnew, lambda, pos] = monqp(H,c,A,b,C,l,verbose,X,ps,xinit)
    % min 1/2  x' H x - c' x
    %  x
    % contrainte   A' x = b
    %
    % et         0 <= x_i  <= C_i
    H = G ;
    n = size(G,1);
    m = size(yTest, 2)
    c = ones(n,1);
    yApp=double(yApp);
    yTest = double(yTest) ;
    A = yApp';
    b = 0 ;
    C = 0.75;
    l = 10^-5;
    verbose = 0;

    [alpha, lambda, pos,mu]= monqp(H,c,A,b,C*ones(n,1),l,verbose) ;
    b = lambda ;
    % alpha ne contient que les indices des vecteurs supports
    % -> alpha sa reconstruction
    w = zeros(n,1) ;
    w(pos) = alpha ;
    
    % Calcul des prédictions
    yPred = K*(yApp'.*w) + b ;

    % Evaluation sur les données d'apprentissage
    aApp = yApp'.*yPred(1:n) ;
    nbPCApp = size(aApp(aApp>0),1) % Nombre de "prédictions" correctes
    accApp = nbPCApp/n % Taux de "prédictions" correctes
   
    % [X,Y,~,AUC,OPTROCPT] = perfcurve(yApp',yPred(1:n),1) ;
    % figure(1),
    % plot(X,Y);
    % title(['ROC curve for Fisher kernel obtained AUC on training data: ', num2str(AUC)]);
   
    % Evaluation sur les données de test
    aTest = yTest'.*yPred((n+1):(n+m)) ;
    nbPCTest = size(aTest(aTest>0),1) % Nombre de prédictions correctes
    accTest = nbPCTest/m % Taux de prédictions correctes

    % [Xt,Yt,~,AUCt,OPTROCPTt] = perfcurve(yTest',yPred((n+1):(n+m)),1) ;
    % figure(2),
    % plot(Xt,Yt);
    % title(['ROC curve for Fisher kernel obtained AUC on test data: ', num2str(AUCt)]);

    %Ecriture des résultats dans le fichier
    save(outFile, 'w', 'b', 'accTest')

    exit
end    
