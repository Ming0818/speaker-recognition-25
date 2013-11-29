function cepstraux(wavfile, matfile)
  s=wavread(wavfile);
  c=melcepst(s, 11025, 'e0dD');
  save(matfile,'c', '-mat-binary');  
  exit
end
