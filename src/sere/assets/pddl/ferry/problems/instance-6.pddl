(define (problem ferry-6)
  (:domain ferry)
  (:objects
    a b c d - location
    c1 c2 c3 - car
  )
  (:init
    (not-eq a b) (not-eq b a) (not-eq a c) (not-eq c a)
    (not-eq a d) (not-eq d a) (not-eq b c) (not-eq c b)
    (not-eq b d) (not-eq d b) (not-eq c d) (not-eq d c)
    (at-ferry a)
    (at c1 a) (at c2 b) (at c3 c)
    (empty-ferry)
  )
  (:goal (and (at c1 d) (at c2 d) (at c3 d)))
)
