import unittest
from h2_word_events import next_word,needs_boundary,boundary_flags,first_character,byte_event_map,is_ascii_atom


class WordEventTests(unittest.TestCase):
    def test_atomic_digits_and_punctuation(self):
        self.assertEqual(next_word(' 12'), ' 1')
        self.assertEqual(next_word('.abc'), '.')
        self.assertFalse(needs_boundary(' 1'))
        self.assertFalse(needs_boundary('.'))

    def test_letters_require_completion(self):
        self.assertIsNone(next_word('the'))
        self.assertEqual(next_word(' then.'),' then')
        self.assertEqual(next_word('the',terminated=True),'the')
        self.assertTrue(needs_boundary('the'))

    def test_latex_and_backslash(self):
        self.assertIsNone(next_word('\\'))
        self.assertIsNone(next_word(r'\frac'))
        self.assertEqual(next_word(r'\frac{'),r'\frac')
        self.assertEqual(next_word('\\.'),'\\')
        self.assertTrue(needs_boundary('\\'))

    def test_unicode_grammar_agrees_with_boundary(self):
        for symbol in ('Ⅳ','²','_','😀'):
            self.assertEqual(next_word(symbol+'A'),symbol)
            self.assertFalse(needs_boundary(symbol))
            self.assertTrue(boundary_flags([symbol.encode()],1,set())[0])
        self.assertIsNone(next_word('éA'))
        self.assertFalse(boundary_flags(['é'.encode()],1,set())[0])
        self.assertTrue(boundary_flags(['é'.encode()],1,set(),latex=True)[0])

    def test_partial_utf8_does_not_hide_leading_punctuation(self):
        self.assertEqual(first_character(b'.\xe2'),'.')
        self.assertEqual(first_character(b'\xe2'),'')
        self.assertTrue(boundary_flags([b'.\xe2',None],3,{1})[0])
        self.assertTrue(boundary_flags([b'.\xe2',None],3,{1})[1])
        self.assertFalse(boundary_flags([b'.\xe2',None],3,{1})[2])

    def test_atomic_marginal_includes_crossing_tokens_and_preserves_mass(self):
        mapping=byte_event_map([b'1',b'123',b'9',b'.',b'...',None],7,{5})
        self.assertEqual(mapping,[49,49,57,46,46,256,257])
        probabilities=[.1,.2,.1,.1,.05,.3,.15]; projected=[0.]*258
        for i,p in zip(mapping,probabilities): projected[i]+=p
        self.assertAlmostEqual(sum(projected),1.)
        self.assertAlmostEqual(projected[49],.3)
        self.assertAlmostEqual(projected[46],.15)
        for value in ('1','.','_'): self.assertTrue(is_ascii_atom(value))
        for value in ('\\','A',' ','Ⅳ','😀','12'): self.assertFalse(is_ascii_atom(value))
