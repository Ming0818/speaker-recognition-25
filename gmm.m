function gmm(mfccfile,matfile)
  load(mfccfile);
  [mu, sig, pi] = gaussmix(c, [], [], 4);
  save(matfile, "mu", "sig", "pi", "-mat-binary");
  exit
end
