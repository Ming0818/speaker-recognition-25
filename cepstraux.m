function cepstraux(wavfile, matfile)
  [s,fs]=wavread(wavfile);
c=melcepst(s, fs, 'e0dD', 16);
  save(matfile, "c", "-mat-binary");  
  exit
end
