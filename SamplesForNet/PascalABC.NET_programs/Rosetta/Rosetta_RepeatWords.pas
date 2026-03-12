// https://rosettacode.org/wiki/Reverse_words_in_a_string#PascalABC.NET

begin
  var text := '''
    ---------- Ice and Fire ------------
    
    fire, in end will world the say Some
    ice. in say Some
    desire of tasted I've what From
    fire. favor who those with hold I
    
    ... elided paragraph last ...
    
    Frost Robert -----------------------
    ''';
  text.ToLines.Select(line -> line.ToWords.Reverse.JoinToString).PrintLines
end.