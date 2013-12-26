function cepstraux(wavfile, matfile)
  [s,fs]=wavread(wavfile);
  c=melcepst(s, fs, 'e0dD');
  #c=melcepst(s, fs);
  save(matfile, "c", "-mat-binary");  
  exit
end
