// https://rosettacode.org/wiki/Count_occurrences_of_a_substring#PascalABC.NET

function CountSubstrings(s,subs: string): integer := s.IndicesOf(subs).Count;

begin
  Print(CountSubstrings('the three truths','th'));
  Print(CountSubstrings('ababababab','abab'));
end.