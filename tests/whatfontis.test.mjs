import test from 'node:test'
import assert from 'node:assert/strict'
import { family, adobeFamily, tokens } from '../scripts/whatfontis.mjs'

test('an Adobe title finds its family through the store words around it, within Adobe alone', () => {
  const adobe = ['New Atten Round', 'ITC American Typewriter', 'Vista Slab', 'OCR A', 'ATF Alternate Gothic', 'DIN 1451 Pro', 'Neue Haas Grotesk', 'New Hero', 'Trajan', 'Dico Typewriter', 'Dico Slab', 'Hustle', 'Winter', 'Nova Mono', 'NovaMono']
  for (const [title, expected] of [['Atten Round New Medium', 'New Atten Round'], ['American Typewriter ITC Pro Condensed', 'ITC American Typewriter'], ['Vista Slab OTCE Reg', 'Vista Slab'],
    ['OCR A Std Regular', 'OCR A'], ['Alternate Gothic ATF Regular', 'ATF Alternate Gothic'], ['DIN 1451 LT Pro Engschrift', 'DIN 1451 Pro'], ['Neue Haas Grotesk Text Pro 65 Medium', 'Neue Haas Grotesk'],
    ['Hero New Regular', 'New Hero'], ['Trajan Pro 3 Regular', 'Trajan']]) assert.equal(adobeFamily(title, adobe), expected, title)
  assert.equal(adobeFamily('Dico Typewriter Slab', adobe), null)   // a tie between Dico Typewriter and Dico Slab names none
  assert.equal(adobeFamily('Hustle Kindness', adobe), null)        // Kindness is no store word: not Hustle
  assert.equal(adobeFamily('Winter Unicorn', adobe), null)
  assert.equal(adobeFamily('Billysans', adobe), null)
  assert.deepEqual(tokens("Haboro Soft Norm ExBold"), ['haboro', 'soft', 'normal', 'extra', 'bold'])
  assert.deepEqual(tokens("BokutohRera 5"), ['bokutoh', 'rera', '5'])
  assert.equal(family('Futura PT Book'), 'Futura PT')
})
