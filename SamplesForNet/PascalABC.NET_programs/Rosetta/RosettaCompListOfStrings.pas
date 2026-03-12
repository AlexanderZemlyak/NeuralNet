// https://rosettacode.org/wiki/Compare_a_list_of_strings

function AllEqual(lst: sequence of string): boolean
  := lst.All(x -> x = lst.First);
  
function IsSorted(lst: sequence of string): boolean
  := lst.Order.SequenceEqual(lst);

function IsStrictAscending(lst: sequence of string): boolean
  := lst.Pairwise.All(x -> x[0] < x[1]);

begin
  var strings := |'abc','abc','abc','abc'|;
  Print(AllEqual(strings));
  var strings1 := |'abc','abd','ade','aef'|;
  Print(IsStrictAscending(strings1));
end.