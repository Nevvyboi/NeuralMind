#!/usr/bin/env python3
"""
GroundZero AI - Advanced NLP Extraction Module
===============================================

State-of-the-art fact and causal relation extraction using:
- spaCy for dependency parsing, POS tagging, NER
- Subject-Verb-Object (SVO) triple extraction
- Linguistic pattern-based causal relation extraction
- Named Entity Recognition for proper entity extraction
- Compound noun handling for complete entities

Based on research from:
- textacy SVO extraction
- spaCy dependency parsing
- Causal relation extraction surveys (Yang et al., 2021)

Author: GroundZero AI
Version: 1.0.0
"""

import re
from typing import List, Tuple, Dict, Optional, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EntityType(Enum):
    """Named entity types from spaCy"""
    PERSON = auto()
    ORG = auto()
    GPE = auto()  # Geopolitical entity
    LOC = auto()
    DATE = auto()
    EVENT = auto()
    PRODUCT = auto()
    WORK_OF_ART = auto()
    CONCEPT = auto()  # Custom: general concepts
    UNKNOWN = auto()


@dataclass
class Entity:
    """Represents an extracted entity"""
    Text: str
    Type: EntityType
    StartChar: int = 0
    EndChar: int = 0
    
    def __hash__(self):
        return hash(self.Text.lower())
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.Text.lower() == other.Text.lower()
        return False


@dataclass
class Triple:
    """Subject-Predicate-Object triple"""
    Subject: str
    Predicate: str
    Object: str
    Confidence: float = 1.0
    SourceSentence: str = ""
    SubjectEntity: Optional[Entity] = None
    ObjectEntity: Optional[Entity] = None
    
    def ToTuple(self) -> Tuple[str, str, str]:
        return (self.Subject, self.Predicate, self.Object)
    
    def __hash__(self):
        return hash((self.Subject.lower(), self.Predicate.lower(), self.Object.lower()))


@dataclass 
class CausalRelation:
    """Cause-Effect relation with metadata"""
    Cause: str
    Effect: str
    Strength: float = 0.8
    Pattern: str = ""  # Which pattern matched
    SourceSentence: str = ""
    CauseEntity: Optional[Entity] = None
    EffectEntity: Optional[Entity] = None
    
    def __hash__(self):
        return hash((self.Cause.lower(), self.Effect.lower()))


@dataclass
class ExtractionResult:
    """Complete extraction result from text"""
    Triples: List[Triple] = field(default_factory=list)
    CausalRelations: List[CausalRelation] = field(default_factory=list)
    Entities: List[Entity] = field(default_factory=list)
    SourceText: str = ""
    ProcessingTimeMs: int = 0


# =============================================================================
# SPACY NLP EXTRACTOR
# =============================================================================

class SpacyNLPExtractor:
    """
    State-of-the-art NLP extractor using spaCy
    
    Features:
    - Dependency parsing for accurate SVO extraction
    - Named Entity Recognition
    - Compound noun handling
    - Causal pattern matching with linguistic validation
    """
    
    # Dependency tags for subjects
    SUBJECT_DEPS = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'expl'}
    
    # Dependency tags for objects
    OBJECT_DEPS = {'dobj', 'pobj', 'dative', 'attr', 'oprd', 'obj'}
    
    # Causal verbs - explicit causation markers
    CAUSAL_VERBS = {
        'cause', 'causes', 'caused', 'causing',
        'lead', 'leads', 'led', 'leading',
        'result', 'results', 'resulted', 'resulting',
        'produce', 'produces', 'produced', 'producing',
        'create', 'creates', 'created', 'creating',
        'trigger', 'triggers', 'triggered', 'triggering',
        'induce', 'induces', 'induced', 'inducing',
        'generate', 'generates', 'generated', 'generating',
        'bring', 'brings', 'brought', 'bringing',
        'make', 'makes', 'made', 'making',
        'affect', 'affects', 'affected', 'affecting',
        'influence', 'influences', 'influenced', 'influencing',
        'determine', 'determines', 'determined', 'determining',
        'contribute', 'contributes', 'contributed', 'contributing',
        'prevent', 'prevents', 'prevented', 'preventing',
        'enable', 'enables', 'enabled', 'enabling',
        'allow', 'allows', 'allowed', 'allowing',
        'force', 'forces', 'forced', 'forcing',
    }
    
    # Causal connectives
    CAUSAL_CONNECTIVES = {
        'because', 'since', 'therefore', 'thus', 'hence',
        'consequently', 'accordingly', 'so', 'as',
        'due to', 'owing to', 'thanks to', 'as a result',
        'for this reason', 'on account of', 'by virtue of'
    }
    
    # Causal patterns (regex-based with linguistic validation)
    CAUSAL_PATTERNS = [
        # X causes Y
        (r'(.+?)\s+(?:causes?|caused)\s+(.+)', 'cause_verb'),
        # X leads to Y
        (r'(.+?)\s+(?:leads?\s+to|led\s+to)\s+(.+)', 'lead_to'),
        # X results in Y
        (r'(.+?)\s+(?:results?\s+in|resulted\s+in)\s+(.+)', 'result_in'),
        # Because of X, Y / Y because of X
        (r'because\s+of\s+(.+?),\s*(.+)', 'because_of'),
        (r'(.+?)\s+because\s+of\s+(.+)', 'because_of_reverse'),
        # Due to X, Y
        (r'due\s+to\s+(.+?),\s*(.+)', 'due_to'),
        # X contributes to Y
        (r'(.+?)\s+(?:contributes?\s+to|contributed\s+to)\s+(.+)', 'contribute_to'),
        # X is caused by Y
        (r'(.+?)\s+(?:is|are|was|were)\s+caused\s+by\s+(.+)', 'caused_by'),
        # X triggers Y
        (r'(.+?)\s+(?:triggers?|triggered)\s+(.+)', 'trigger'),
        # X produces Y
        (r'(.+?)\s+(?:produces?|produced)\s+(.+)', 'produce'),
        # X affects Y
        (r'(.+?)\s+(?:affects?|affected)\s+(.+)', 'affect'),
        # X influences Y
        (r'(.+?)\s+(?:influences?|influenced)\s+(.+)', 'influence'),
        # If X then Y
        (r'if\s+(.+?),?\s+then\s+(.+)', 'if_then'),
        # When X, Y happens
        (r'when\s+(.+?),\s*(.+?)(?:\s+happens?|\s+occurs?)?', 'when_then'),
        # X makes Y happen
        (r'(.+?)\s+(?:makes?|made)\s+(.+?)\s+(?:happen|occur)', 'make_happen'),
        # X prevents Y
        (r'(.+?)\s+(?:prevents?|prevented)\s+(.+)', 'prevent'),
        # X enables Y
        (r'(.+?)\s+(?:enables?|enabled)\s+(.+)', 'enable'),
        # As a result of X, Y
        (r'as\s+a\s+result\s+of\s+(.+?),\s*(.+)', 'as_result'),
        # Consequently, / Therefore,
        (r'(.+?)[.;]\s*(?:consequently|therefore|thus|hence),?\s*(.+)', 'consequence'),
    ]
    
    # Relation verbs for SVO extraction (predicate types)
    RELATION_VERBS = {
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'having',
        'includes', 'include', 'included',
        'contains', 'contain', 'contained',
        'consists', 'consist', 'consisted',
        'comprises', 'comprise', 'comprised',
        'means', 'mean', 'meant',
        'represents', 'represent', 'represented',
        'refers', 'refer', 'referred',
        'defines', 'define', 'defined',
        'describes', 'describe', 'described',
        'involves', 'involve', 'involved',
        'requires', 'require', 'required',
        'uses', 'use', 'used',
        'provides', 'provide', 'provided',
        'supports', 'support', 'supported',
        'helps', 'help', 'helped',
        'allows', 'allow', 'allowed',
        'enables', 'enable', 'enabled',
        'creates', 'create', 'created',
        'produces', 'produce', 'produced',
        'makes', 'make', 'made',
        'forms', 'form', 'formed',
        'becomes', 'become', 'became',
        'remains', 'remain', 'remained',
        'exists', 'exist', 'existed',
        'occurs', 'occur', 'occurred',
        'happens', 'happen', 'happened',
        'develops', 'develop', 'developed',
        'grows', 'grow', 'grew',
        'increases', 'increase', 'increased',
        'decreases', 'decrease', 'decreased',
        'changes', 'change', 'changed',
        'affects', 'affect', 'affected',
        'influences', 'influence', 'influenced',
        'determines', 'determine', 'determined',
        'controls', 'control', 'controlled',
        'regulates', 'regulate', 'regulated',
        'manages', 'manage', 'managed',
        'operates', 'operate', 'operated',
        'works', 'work', 'worked',
        'functions', 'function', 'functioned',
        'serves', 'serve', 'served',
        'acts', 'act', 'acted',
        'plays', 'play', 'played',
        'performs', 'perform', 'performed',
        'runs', 'run', 'ran',
        'moves', 'move', 'moved',
        'travels', 'travel', 'traveled',
        'flows', 'flow', 'flowed',
        'spreads', 'spread',
        'covers', 'cover', 'covered',
        'surrounds', 'surround', 'surrounded',
        'connects', 'connect', 'connected',
        'links', 'link', 'linked',
        'joins', 'join', 'joined',
        'combines', 'combine', 'combined',
        'separates', 'separate', 'separated',
        'divides', 'divide', 'divided',
        'breaks', 'break', 'broke',
        'destroys', 'destroy', 'destroyed',
        'kills', 'kill', 'killed',
        'dies', 'die', 'died',
        'lives', 'live', 'lived',
        'born', 'bears', 'bear',
        'eats', 'eat', 'ate',
        'drinks', 'drink', 'drank',
        'breathes', 'breathe', 'breathed',
        'sleeps', 'sleep', 'slept',
        'wakes', 'wake', 'woke',
        'sees', 'see', 'saw',
        'hears', 'hear', 'heard',
        'feels', 'feel', 'felt',
        'thinks', 'think', 'thought',
        'knows', 'know', 'knew',
        'believes', 'believe', 'believed',
        'understands', 'understand', 'understood',
        'learns', 'learn', 'learned',
        'teaches', 'teach', 'taught',
        'studies', 'study', 'studied',
        'researches', 'research', 'researched',
        'discovers', 'discover', 'discovered',
        'finds', 'find', 'found',
        'invents', 'invent', 'invented',
        'designs', 'design', 'designed',
        'builds', 'build', 'built',
        'constructs', 'construct', 'constructed',
        'writes', 'write', 'wrote',
        'reads', 'read',
        'speaks', 'speak', 'spoke',
        'says', 'say', 'said',
        'tells', 'tell', 'told',
        'asks', 'ask', 'asked',
        'answers', 'answer', 'answered',
        'explains', 'explain', 'explained',
        'shows', 'show', 'showed',
        'demonstrates', 'demonstrate', 'demonstrated',
        'proves', 'prove', 'proved',
        'tests', 'test', 'tested',
        'measures', 'measure', 'measured',
        'calculates', 'calculate', 'calculated',
        'computes', 'compute', 'computed',
        'processes', 'process', 'processed',
        'stores', 'store', 'stored',
        'saves', 'save', 'saved',
        'loads', 'load', 'loaded',
        'sends', 'send', 'sent',
        'receives', 'receive', 'received',
        'transmits', 'transmit', 'transmitted',
        'emits', 'emit', 'emitted',
        'absorbs', 'absorb', 'absorbed',
        'reflects', 'reflect', 'reflected',
        'refracts', 'refract', 'refracted',
        'rotates', 'rotate', 'rotated',
        'spins', 'spin', 'spun',
        'orbits', 'orbit', 'orbited',
        'attracts', 'attract', 'attracted',
        'repels', 'repel', 'repelled',
        'pulls', 'pull', 'pulled',
        'pushes', 'push', 'pushed',
        'holds', 'hold', 'held',
        'releases', 'release', 'released',
        'drops', 'drop', 'dropped',
        'rises', 'rise', 'rose',
        'falls', 'fall', 'fell',
        'floats', 'float', 'floated',
        'sinks', 'sink', 'sank',
        'melts', 'melt', 'melted',
        'freezes', 'freeze', 'froze',
        'boils', 'boil', 'boiled',
        'evaporates', 'evaporate', 'evaporated',
        'condenses', 'condense', 'condensed',
        'dissolves', 'dissolve', 'dissolved',
        'mixes', 'mix', 'mixed',
        'reacts', 'react', 'reacted',
        'burns', 'burn', 'burned',
        'explodes', 'explode', 'exploded',
        'expands', 'expand', 'expanded',
        'contracts', 'contract', 'contracted',
        'vibrates', 'vibrate', 'vibrated',
        'oscillates', 'oscillate', 'oscillated',
        'resonates', 'resonate', 'resonated',
    }
    
    # Words to filter out from subjects/objects
    STOPWORDS = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'it', 'its', 'they', 'them', 'their', 'he', 'she',
        'him', 'her', 'his', 'hers', 'we', 'us', 'our',
        'i', 'me', 'my', 'you', 'your', 'one', 'ones',
        'some', 'any', 'all', 'most', 'many', 'few',
        'much', 'more', 'less', 'other', 'another',
        'such', 'same', 'different', 'various', 'several',
        'each', 'every', 'both', 'either', 'neither',
        'no', 'none', 'nothing', 'something', 'anything',
        'everything', 'someone', 'anyone', 'everyone',
        'nobody', 'somebody', 'anybody', 'everybody',
        'who', 'what', 'which', 'where', 'when', 'why', 'how',
        'very', 'really', 'quite', 'rather', 'too', 'also',
        'just', 'only', 'even', 'still', 'already', 'yet',
        'now', 'then', 'here', 'there', 'always', 'never',
        'often', 'sometimes', 'usually', 'generally',
        'however', 'therefore', 'thus', 'hence', 'moreover',
        'furthermore', 'additionally', 'meanwhile', 'otherwise',
        'instead', 'rather', 'although', 'though', 'unless',
        'whether', 'while', 'whereas', 'whenever', 'wherever',
        'whatever', 'whoever', 'whichever', 'however',
        'etc', 'eg', 'ie', 'vs', 'etc.', 'e.g.', 'i.e.',
    }
    
    def __init__(self, ModelName: str = "en_core_web_sm"):
        """
        Initialize the NLP extractor
        
        Args:
            ModelName: spaCy model to use (sm, md, lg, trf)
        """
        self.ModelName = ModelName
        self.NLP = None
        self._LoadModel()
    
    def _LoadModel(self):
        """Load spaCy model, download if not available"""
        try:
            import spacy
            try:
                self.NLP = spacy.load(self.ModelName)
                print(f"✓ Loaded spaCy model: {self.ModelName}")
            except OSError:
                print(f"⬇️ Downloading spaCy model: {self.ModelName}...")
                from spacy.cli import download
                download(self.ModelName)
                self.NLP = spacy.load(self.ModelName)
                print(f"✓ Downloaded and loaded: {self.ModelName}")
        except ImportError:
            print("⚠️ spaCy not installed. Using fallback extraction.")
            self.NLP = None
    
    def IsAvailable(self) -> bool:
        """Check if spaCy is available"""
        return self.NLP is not None
    
    def Extract(self, Text: str) -> ExtractionResult:
        """
        Extract all facts and causal relations from text
        
        Args:
            Text: Input text to process
            
        Returns:
            ExtractionResult with triples, causal relations, entities
        """
        import time
        StartTime = time.time()
        
        Result = ExtractionResult(SourceText=Text)
        
        if not self.NLP:
            # Fallback to regex-based extraction
            Result.Triples = self._FallbackExtractTriples(Text)
            Result.CausalRelations = self._FallbackExtractCausal(Text)
        else:
            # Use spaCy for extraction
            Doc = self.NLP(Text)
            
            # Extract entities
            Result.Entities = self._ExtractEntities(Doc)
            
            # Extract SVO triples
            Result.Triples = self._ExtractSVOTriples(Doc)
            
            # Extract causal relations
            Result.CausalRelations = self._ExtractCausalRelations(Doc)
        
        # Deduplicate
        Result.Triples = list(set(Result.Triples))
        Result.CausalRelations = list(set(Result.CausalRelations))
        
        Result.ProcessingTimeMs = int((time.time() - StartTime) * 1000)
        
        return Result
    
    def _ExtractEntities(self, Doc) -> List[Entity]:
        """Extract named entities from spaCy doc"""
        Entities = []
        
        for Ent in Doc.ents:
            EntType = EntityType.UNKNOWN
            try:
                EntType = EntityType[Ent.label_]
            except KeyError:
                if Ent.label_ in ('NORP', 'FAC', 'LAW', 'LANGUAGE'):
                    EntType = EntityType.CONCEPT
            
            Entities.append(Entity(
                Text=Ent.text,
                Type=EntType,
                StartChar=Ent.start_char,
                EndChar=Ent.end_char
            ))
        
        return Entities
    
    def _ExtractSVOTriples(self, Doc) -> List[Triple]:
        """
        Extract Subject-Verb-Object triples using dependency parsing
        
        This is the core improvement over regex - using actual linguistic structure
        """
        Triples = []
        
        for Sent in Doc.sents:
            # Find the root verb
            Root = None
            for Token in Sent:
                if Token.dep_ == 'ROOT' and Token.pos_ == 'VERB':
                    Root = Token
                    break
            
            if not Root:
                # Try to find any verb as root
                for Token in Sent:
                    if Token.pos_ == 'VERB':
                        Root = Token
                        break
            
            if not Root:
                continue
            
            # Extract triples from this sentence
            SentTriples = self._ExtractTriplesFromVerb(Root, Sent)
            Triples.extend(SentTriples)
            
            # Also check other verbs in the sentence
            for Token in Sent:
                if Token.pos_ == 'VERB' and Token != Root:
                    OtherTriples = self._ExtractTriplesFromVerb(Token, Sent)
                    Triples.extend(OtherTriples)
        
        return Triples
    
    def _ExtractTriplesFromVerb(self, Verb, Sent) -> List[Triple]:
        """Extract triples with a specific verb as predicate"""
        Triples = []
        
        # Find subjects
        Subjects = []
        for Child in Verb.children:
            if Child.dep_ in self.SUBJECT_DEPS:
                SubjPhrase = self._GetFullPhrase(Child)
                if self._IsValidEntity(SubjPhrase):
                    Subjects.append(SubjPhrase)
        
        # Find objects
        Objects = []
        for Child in Verb.children:
            if Child.dep_ in self.OBJECT_DEPS:
                ObjPhrase = self._GetFullPhrase(Child)
                if self._IsValidEntity(ObjPhrase):
                    Objects.append(ObjPhrase)
            
            # Check for prepositional objects
            elif Child.dep_ == 'prep':
                for PrepChild in Child.children:
                    if PrepChild.dep_ == 'pobj':
                        ObjPhrase = self._GetFullPhrase(PrepChild)
                        if self._IsValidEntity(ObjPhrase):
                            # Include preposition in relation
                            PrepPhrase = f"{Verb.lemma_} {Child.text}"
                            Objects.append((ObjPhrase, PrepPhrase))
        
        # Also check for attributes (for "is a" type relations)
        for Child in Verb.children:
            if Child.dep_ == 'attr':
                ObjPhrase = self._GetFullPhrase(Child)
                if self._IsValidEntity(ObjPhrase):
                    Objects.append(ObjPhrase)
        
        # Generate triples
        VerbLemma = self._NormalizePredicate(Verb)
        
        for Subj in Subjects:
            for Obj in Objects:
                if isinstance(Obj, tuple):
                    ObjText, Pred = Obj
                    Triple_ = Triple(
                        Subject=self._CleanEntity(Subj),
                        Predicate=Pred,
                        Object=self._CleanEntity(ObjText),
                        Confidence=0.9,
                        SourceSentence=Sent.text
                    )
                else:
                    Triple_ = Triple(
                        Subject=self._CleanEntity(Subj),
                        Predicate=VerbLemma,
                        Object=self._CleanEntity(Obj),
                        Confidence=0.9,
                        SourceSentence=Sent.text
                    )
                
                if Triple_.Subject and Triple_.Object and Triple_.Subject != Triple_.Object:
                    Triples.append(Triple_)
        
        return Triples
    
    def _GetFullPhrase(self, Token) -> str:
        """
        Get the full noun phrase including compounds and modifiers
        
        This is crucial for getting "Gordian Capital" instead of just "Capital"
        """
        # Get all tokens in subtree
        Subtree = list(Token.subtree)
        
        # Sort by position
        Subtree.sort(key=lambda t: t.i)
        
        # Filter to keep relevant parts
        RelevantDeps = {'compound', 'amod', 'nmod', 'nummod', 'det', 'poss', 'case'}
        
        Parts = []
        for T in Subtree:
            # Include the head token
            if T == Token:
                Parts.append(T.text)
            # Include compounds and modifiers
            elif T.dep_ in RelevantDeps or T.head == Token:
                Parts.append(T.text)
            # Include conjuncts
            elif T.dep_ == 'conj' and T.head == Token:
                continue  # Handle separately
        
        # Reconstruct phrase
        Phrase = ' '.join(Parts)
        
        return Phrase
    
    def _NormalizePredicate(self, Verb) -> str:
        """Normalize verb to base form with any particles"""
        Parts = [Verb.lemma_]
        
        # Check for verb particles
        for Child in Verb.children:
            if Child.dep_ == 'prt':  # Particle
                Parts.append(Child.text)
        
        return '_'.join(Parts)
    
    def _IsValidEntity(self, Text: str) -> bool:
        """Check if text is a valid entity (not just stopwords)"""
        if not Text:
            return False
        
        Words = Text.lower().split()
        
        # Filter stopwords
        ContentWords = [W for W in Words if W not in self.STOPWORDS]
        
        # Must have at least one content word
        if not ContentWords:
            return False
        
        # Must have at least 2 characters
        if len(''.join(ContentWords)) < 2:
            return False
        
        return True
    
    def _CleanEntity(self, Text: str) -> str:
        """Clean and normalize entity text"""
        # Remove leading/trailing stopwords
        Words = Text.split()
        
        while Words and Words[0].lower() in self.STOPWORDS:
            Words.pop(0)
        
        while Words and Words[-1].lower() in self.STOPWORDS:
            Words.pop()
        
        Text = ' '.join(Words)
        
        # Normalize whitespace
        Text = ' '.join(Text.split())
        
        # Remove quotes
        Text = Text.strip('"\'')
        
        # Title case for proper nouns
        if Text and Text[0].isupper():
            return Text
        
        return Text.lower()
    
    def _ExtractCausalRelations(self, Doc) -> List[CausalRelation]:
        """
        Extract causal relations using linguistic patterns and dependency parsing
        """
        Relations = []
        
        for Sent in Doc.sents:
            SentText = Sent.text
            
            # Method 1: Causal verb detection
            for Token in Sent:
                if Token.lemma_ in {'cause', 'lead', 'result', 'trigger', 'produce', 'create', 'induce'}:
                    CausalRel = self._ExtractCausalFromVerb(Token, Sent)
                    if CausalRel:
                        Relations.append(CausalRel)
            
            # Method 2: Pattern matching with linguistic validation
            for Pattern, PatternName in self.CAUSAL_PATTERNS:
                Match = re.search(Pattern, SentText, re.IGNORECASE)
                if Match:
                    Cause = Match.group(1).strip()
                    Effect = Match.group(2).strip()
                    
                    # Validate with NLP
                    if self._ValidateCausalRelation(Cause, Effect, Doc):
                        # Clean and extract main concepts
                        CauseClean = self._ExtractMainConcept(Cause, Doc)
                        EffectClean = self._ExtractMainConcept(Effect, Doc)
                        
                        if CauseClean and EffectClean and CauseClean != EffectClean:
                            Relations.append(CausalRelation(
                                Cause=CauseClean,
                                Effect=EffectClean,
                                Strength=0.8,
                                Pattern=PatternName,
                                SourceSentence=SentText
                            ))
        
        return Relations
    
    def _ExtractCausalFromVerb(self, Verb, Sent) -> Optional[CausalRelation]:
        """Extract causal relation from a causal verb"""
        Cause = None
        Effect = None
        
        for Child in Verb.children:
            # Subject is usually the cause
            if Child.dep_ in self.SUBJECT_DEPS:
                Cause = self._GetFullPhrase(Child)
            
            # Object is usually the effect
            elif Child.dep_ in self.OBJECT_DEPS:
                Effect = self._GetFullPhrase(Child)
            
            # Handle "leads to X"
            elif Child.dep_ == 'prep' and Child.text.lower() in ('to', 'in'):
                for PrepChild in Child.children:
                    if PrepChild.dep_ == 'pobj':
                        Effect = self._GetFullPhrase(PrepChild)
        
        if Cause and Effect:
            CauseClean = self._CleanEntity(Cause)
            EffectClean = self._CleanEntity(Effect)
            
            if CauseClean and EffectClean and CauseClean != EffectClean:
                return CausalRelation(
                    Cause=CauseClean,
                    Effect=EffectClean,
                    Strength=0.85,
                    Pattern=f"verb_{Verb.lemma_}",
                    SourceSentence=Sent.text
                )
        
        return None
    
    def _ValidateCausalRelation(self, Cause: str, Effect: str, Doc) -> bool:
        """Validate that the extracted cause/effect are valid noun phrases"""
        # Basic length check
        if len(Cause) < 2 or len(Effect) < 2:
            return False
        
        # Check they're not the same
        if Cause.lower() == Effect.lower():
            return False
        
        # Check they have content words
        if not self._IsValidEntity(Cause) or not self._IsValidEntity(Effect):
            return False
        
        return True
    
    def _ExtractMainConcept(self, Text: str, Doc) -> str:
        """Extract the main concept (noun phrase) from text"""
        # Process with spaCy
        TextDoc = self.NLP(Text)
        
        # Find noun chunks
        Chunks = list(TextDoc.noun_chunks)
        
        if Chunks:
            # Return the largest/most important chunk
            BestChunk = max(Chunks, key=lambda c: len(c.text))
            return self._CleanEntity(BestChunk.text)
        
        # Fallback: find nouns
        Nouns = [T for T in TextDoc if T.pos_ in ('NOUN', 'PROPN')]
        
        if Nouns:
            # Get the last noun (usually most specific)
            MainNoun = Nouns[-1]
            return self._CleanEntity(self._GetFullPhrase(MainNoun))
        
        # Last resort: clean the whole text
        return self._CleanEntity(Text)
    
    # =========================================================================
    # FALLBACK METHODS (when spaCy not available)
    # =========================================================================
    
    def _FallbackExtractTriples(self, Text: str) -> List[Triple]:
        """Regex-based triple extraction (fallback)"""
        Triples = []
        
        Patterns = [
            # X is a Y
            (r'([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)\s+is\s+(?:a|an|the)\s+([a-z]+(?:\s+[a-z]+)?)', 'is_a'),
            # X are Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+are\s+([a-z]+(?:\s+[a-z]+)?)', 'are'),
            # X has Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+(?:has|have)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)', 'has'),
            # X contains Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+contains?\s+([a-z]+(?:\s+[a-z]+)?)', 'contains'),
            # X consists of Y
            (r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+consists?\s+of\s+([a-z]+(?:\s+[a-z]+)?)', 'consists_of'),
        ]
        
        for Pattern, Pred in Patterns:
            for Match in re.finditer(Pattern, Text):
                Subj = Match.group(1).strip()
                Obj = Match.group(2).strip()
                
                if self._IsValidEntity(Subj) and self._IsValidEntity(Obj):
                    Triples.append(Triple(
                        Subject=self._CleanEntity(Subj),
                        Predicate=Pred,
                        Object=self._CleanEntity(Obj),
                        Confidence=0.7,
                        SourceSentence=Match.group(0)
                    ))
        
        return Triples
    
    def _FallbackExtractCausal(self, Text: str) -> List[CausalRelation]:
        """Regex-based causal extraction (fallback)"""
        Relations = []
        
        for Pattern, PatternName in self.CAUSAL_PATTERNS[:10]:  # Use simpler patterns
            for Match in re.finditer(Pattern, Text, re.IGNORECASE):
                Cause = Match.group(1).strip()
                Effect = Match.group(2).strip()
                
                # Basic cleaning
                Cause = self._CleanEntity(Cause)
                Effect = self._CleanEntity(Effect)
                
                if Cause and Effect and Cause != Effect and len(Cause) > 2 and len(Effect) > 2:
                    Relations.append(CausalRelation(
                        Cause=Cause,
                        Effect=Effect,
                        Strength=0.7,
                        Pattern=PatternName,
                        SourceSentence=Match.group(0)
                    ))
        
        return Relations


# =============================================================================
# INTEGRATION INTERFACE
# =============================================================================

class NLPProcessor:
    """
    High-level interface for NLP processing
    
    Automatically uses spaCy if available, falls back to regex
    """
    
    def __init__(self, UseSpacy: bool = True, Model: str = "en_core_web_sm"):
        """
        Initialize NLP processor
        
        Args:
            UseSpacy: Whether to try using spaCy
            Model: spaCy model name (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        self.Extractor = None
        
        if UseSpacy:
            try:
                self.Extractor = SpacyNLPExtractor(Model)
            except Exception as E:
                print(f"⚠️ Could not initialize spaCy: {E}")
                print("   Using fallback regex extraction")
        
        if not self.Extractor or not self.Extractor.IsAvailable():
            self.Extractor = SpacyNLPExtractor("en_core_web_sm")
    
    def ExtractFacts(self, Text: str) -> List[Tuple[str, str, str]]:
        """
        Extract facts as (subject, predicate, object) tuples
        
        Args:
            Text: Input text
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        Result = self.Extractor.Extract(Text)
        return [T.ToTuple() for T in Result.Triples]
    
    def ExtractCausal(self, Text: str) -> List[Tuple[str, str, float]]:
        """
        Extract causal relations as (cause, effect, strength) tuples
        
        Args:
            Text: Input text
            
        Returns:
            List of (cause, effect, strength) tuples
        """
        Result = self.Extractor.Extract(Text)
        return [(C.Cause, C.Effect, C.Strength) for C in Result.CausalRelations]
    
    def ExtractAll(self, Text: str) -> ExtractionResult:
        """
        Extract all information from text
        
        Args:
            Text: Input text
            
        Returns:
            ExtractionResult with triples, causal relations, entities
        """
        return self.Extractor.Extract(Text)
    
    def IsSpacyAvailable(self) -> bool:
        """Check if spaCy is being used"""
        return self.Extractor.IsAvailable() if self.Extractor else False


# =============================================================================
# TEST / DEMO
# =============================================================================

def Demo():
    """Demonstrate NLP extraction capabilities"""
    print("\n" + "=" * 70)
    print("🧠 GroundZero AI - Advanced NLP Extraction Demo")
    print("=" * 70 + "\n")
    
    # Initialize processor
    Processor = NLPProcessor()
    
    print(f"Using spaCy: {Processor.IsSpacyAvailable()}\n")
    
    # Test texts
    TestTexts = [
        "Physics is the natural science that studies matter and energy. Albert Einstein developed the theory of relativity.",
        "Rain causes wet ground. Wet ground leads to slippery surfaces. Slippery surfaces can cause accidents.",
        "The heart pumps blood through the body. Blood carries oxygen to cells. Lack of oxygen causes cell death.",
        "Climate change is caused by greenhouse gas emissions. Rising temperatures lead to melting ice caps.",
        "Dogs are mammals. Mammals are warm-blooded animals. Dogs have four legs and a tail.",
    ]
    
    for i, Text in enumerate(TestTexts, 1):
        print(f"📝 Text {i}:")
        print(f"   \"{Text[:80]}...\"" if len(Text) > 80 else f"   \"{Text}\"")
        print()
        
        Result = Processor.ExtractAll(Text)
        
        if Result.Entities:
            print(f"   🏷️  Entities: {[E.Text for E in Result.Entities]}")
        
        if Result.Triples:
            print(f"   📚 Facts ({len(Result.Triples)}):")
            for T in Result.Triples[:5]:
                print(f"      • ({T.Subject}, {T.Predicate}, {T.Object})")
        
        if Result.CausalRelations:
            print(f"   🔗 Causal ({len(Result.CausalRelations)}):")
            for C in Result.CausalRelations[:5]:
                print(f"      • {C.Cause} → {C.Effect} ({C.Strength:.0%})")
        
        print(f"   ⏱️  Processed in {Result.ProcessingTimeMs}ms")
        print()
    
    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    Demo()