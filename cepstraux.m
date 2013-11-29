function cepstraux(wavfile, matfile)
  s=wavread(wavfile);
  c=melcepst(s);
  fprintf(matfile);
  save(matfile, 'c');
  exit
end