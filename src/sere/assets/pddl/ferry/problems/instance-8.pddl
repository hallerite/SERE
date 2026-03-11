(define (problem ferry-8)
  (:domain ferry)
  (:objects
    a b c d e - location
    c1 c2 c3 c4 c5 c6 - car
  )
  (:init
    (not-eq a b) (not-eq b a) (not-eq a c) (not-eq c a)
    (not-eq a d) (not-eq d a) (not-eq a e) (not-eq e a)
    (not-eq b c) (not-eq c b) (not-eq b d) (not-eq d b)
    (not-eq b e) (not-eq e b) (not-eq c d) (not-eq d c)
    (not-eq c e) (not-eq e c) (not-eq d e) (not-eq e d)
    (at-ferry a)
    (at c1 a) (at c2 a) (at c3 b) (at c4 c) (at c5 d) (at c6 e)
    (empty-ferry)
  )
  (:goal (and (at c1 e) (at c2 d) (at c3 c) (at c4 b) (at c5 a) (at c6 a)))
)
