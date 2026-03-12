begin
  var suit := '♠♡♢♣';
  var value := |'6','7','8','9','10','В','Д','К','Т'|;
  var deck := suit.Cartesian(value)
    .Select(c -> c[1]+c[0]).ToArray;
  deck.Println;
  Println;
  Shuffle(deck);
  deck.Println;
end.