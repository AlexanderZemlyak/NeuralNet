{ В файле freqs.txt формата 
23 1.53 абрикосовый adj
24 1.35 абсолют noun
25 70.51 абсолютно adv
  выведите 10 первых существительных, содержащих не менее трех букв 'о' }
begin
  // Парсинг текста
  var words := ReadLines('freqs.txt')
    .Select(s -> s.ToWords)
    .Select(w -> new class(freq := w[1], word := w[2], sp := w[3]));
    
  // Собственно решение задачи
  var owords := words.Where(w -> (w.sp = 'noun') and (w.word.CountOf('о') >= 4));

  // Вывод
  owords.Take(15).PrintLines(w -> w.word)  
end.