function cepstraux(wavfile, matfile)
  s=wavread(wavfile);
  c=melcepst(s);
  save(matfile,'c','-mat-binary')  
  exit
end
