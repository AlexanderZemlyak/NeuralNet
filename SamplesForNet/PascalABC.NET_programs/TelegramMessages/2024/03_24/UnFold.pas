/// Получает последовательность из начального значения
function Unfold<T,T1>(initial: T; next: T->(T1,T,boolean)): sequence of T1; 
begin
  var q := next(initial);
  while q[2] do
  begin
    yield q[0];
    initial := q[1];
    q := next(initial);
  end;
end;

/// Получает последовательность из начального значения
function Unfold<T,T1>(initial: T; proj: T->T1; next: T->T; cond: T->boolean): sequence of T1; 
begin
  while cond(initial) do
  begin
    yield proj(initial);
    initial := next(initial);
  end;
end;

/// Получает последовательность из начального значения
function Unfold<T>(initial: T; next: T->T; cond: T->boolean): sequence of T; 
begin
  while cond(initial) do
  begin
    yield initial;
    initial := next(initial);
  end;
end;

/// Получает последовательность из начального значения
function Unfold<T>(initial: T; next: T->(T,boolean)): sequence of T; 
begin
  var q := next(initial);
  while q[1] do
  begin
    yield initial;
    initial := q[0];
    q := next(initial);
  end;
end;

/// Получает последовательность из начального значения
function UnfoldInf<T,T1>(x: T; next: T->(T1,T)): sequence of T1;
begin
  while True do
  begin
    var q := next(x);
    yield q[0];
    x := q[1];
  end;
end;

/// Получает последовательность из начального значения
function UnfoldInf<T,T1>(initial: T; proj: T->T1; next: T->T): sequence of T1;
begin
  while True do
  begin
    yield proj(initial);
    initial := next(initial);
  end;
end;

begin
  var i := 12345;
  Unfold(i, x -> (x div 10,x>0)).Println;
  Unfold(i, x -> x div 10, x -> x>0).Println;
  
  Unfold(i, x -> (x mod 10,x div 10,x>0)).Println;
  Unfold(i, x -> x mod 10, x -> x div 10, x -> x>0).Println;
  
  UnfoldInf((1,1), \(a,b) -> (a,(b,a+b))).Take(10).Println;
  UnfoldInf((1,1), \(a,b) -> a, \(a,b) -> (b,a+b)).Take(10).Println;
end.