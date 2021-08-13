class Display:

    def __init__(self, dummy=True):
        self._dummy = dummy
        if dummy:
            return
        #import curses
        #self._screen = curses.initscr()
        #curses.noecho()
        #curses.curs_set(0)
        #self.set_state(0)

    def set_state(self, state, refresh=True):
        if self._dummy:
            case = {
                2:  'Generate HTML chunks',
                3:  'Detect job posting language',
                4:  'Translate job postings',
                5:  'Parse chunks',
                7:  'Generate NER and POS features',
                8:  'Generate NER and POS dataframes',
                9:  'Generate n-gram features',
                10: 'Classification',
                12: 'Clean chunks',
                13: 'Generate features',
                14: 'Extract job location',
                15: 'Extract educational requirements',
                16: 'Extract employment type',
                17: 'Extract salary information',
                18: 'Clean job postings',
                19: 'Generate features',
                20: 'Extract work hours',
                22: 'Clean job postings',
                23: 'Match job title',
                24: 'Match skills',
                25: 'Filter skills',
                26: 'Match job sector',
                28: 'Generate educational requirements',
                29: 'Generate job location',
                30: 'Generate employment type',
                31: 'Generate base salary',
                32: 'Generate currency info',
                33: 'Clean base salary',
                34: 'Clean employment type',
                35: 'Clean work hours',
                36: 'Clean educational requirements',
                37: 'Merge information',
                39: 'Merge all data',
                40: 'Save data to SQL',
                41: 'Export final data',
            }
            if state in case:
                print(case[state])
            return
        if state == 0:
            self._screen.clear()
            self._screen.addstr(0, 0,  '---------------------------- GENERATE DATA FROM HTML ---------------------------')
            self._screen.addstr(2, 0,  '-------------------------- CLASSIFY USING SAVED MODEL --------------------------')
            self._screen.addstr(4, 0,  '--------------------------------- EXTRACT INFO 1 -------------------------------')
            self._screen.addstr(6, 0,  '--------------------------------- EXTRACT INFO 2 -------------------------------')
            self._screen.addstr(8, 0,  '------------------------------- PROCESS AND MERGE ------------------------------')
            self._screen.addstr(10, 0, '--------------------------------- EXPORT TO SQL --------------------------------')
            self._screen.refresh()

        elif state == 1:
            self.set_state(0, False)
            self._insert_text(1, [
                '  - Generate HTML chunks',
                '  - Detect job posting language',
                '  - Translate job postings',
                '  - Parse chunks',
            ])

        elif state == 2:
            self.set_state(1, False)
            self._display_progr(2, 2)
            self._screen.move(3, 0)
            self._screen.insertln()
            self._screen.move(3, 0)

        elif state == 3:
            self.set_state(1, False)
            self._display_progr(2, 3)
            self._screen.move(4, 0)
            self._screen.insertln()
            self._screen.move(4, 0)

        elif state == 4:
            self.set_state(1, False)
            self._display_progr(2, 4)
            self._screen.move(5, 0)
            self._screen.insertln()
            self._screen.move(5, 0)

        elif state == 5:
            self.set_state(1, False)
            self._display_progr(2, 5)
            self._screen.move(6, 0)
            self._screen.insertln()
            self._screen.move(6, 0)

        elif state == 6:
            self.set_state(0, False)
            self._insert_text(3, [
                '  - Generate NER and POS features',
                '  - Generate NER and POS dataframes',
                '  - Generate n-gram features',
                '  - Classification',
            ])

        elif state == 7:
            self.set_state(6, False)
            self._display_progr(4, 4)
            self._screen.move(5, 0)
            self._screen.insertln()
            self._screen.move(5, 0)

        elif state == 8:
            self.set_state(6, False)
            self._display_progr(4, 5)

        elif state == 9:
            self.set_state(6, False)
            self._display_progr(4, 6)

        elif state == 10:
            self.set_state(6, False)
            self._display_progr(4, 7)
            self._screen.move(8, 0)
            self._screen.insertln()
            self._screen.move(8, 0)

        elif state == 11:
            self.set_state(0, False)
            self._insert_text(5, [
                '  - Clean chunks',
                '  - Generate features',
                '  - Extract job location',
                '  - Extract educational requirements',
                '  - Extract employment type',
                '  - Extract salary information',
                '  - Clean job postings',
                '  - Generate features',
                '  - Extract work hours',
            ])

        elif state == 12:
            self.set_state(11, False)
            self._display_progr(6, 6)
            self._screen.move(7, 0)
            self._screen.insertln()
            self._screen.move(7, 0)

        elif state == 13:
            self.set_state(11, False)
            self._display_progr(6, 7)
            self._screen.move(8, 0)
            self._screen.insertln()
            self._screen.move(8, 0)

        elif state == 14:
            self.set_state(11, False)
            self._display_progr(6, 8)
            self._screen.move(9, 0)
            self._screen.insertln()
            self._screen.move(9, 0)

        elif state == 15:
            self.set_state(11, False)
            self._display_progr(6, 9)

        elif state == 16:
            self.set_state(11, False)
            self._display_progr(6, 10)

        elif state == 17:
            self.set_state(11, False)
            self._display_progr(6, 11)
            self._screen.move(12, 0)
            self._screen.insertln()
            self._screen.move(12, 0)

        elif state == 18:
            self.set_state(11, False)
            self._display_progr(6, 12)
            self._screen.move(13, 0)
            self._screen.insertln()
            self._screen.move(13, 0)

        elif state == 19:
            self.set_state(11, False)
            self._display_progr(6, 13)
            self._screen.move(14, 0)
            self._screen.insertln()
            self._screen.move(14, 0)

        elif state == 20:
            self.set_state(11, False)
            self._display_progr(6, 14)

        elif state == 21:
            self.set_state(0, False)
            self._insert_text(7, [
                '  - Clean job postings',
                '  - Match job title',
                '  - Match skills',
                '  - Filter skills',
                '  - Match job sector',
            ])

        elif state == 22:
            self.set_state(21, False)
            self._display_progr(8, 8)
            self._screen.move(9, 0)
            self._screen.insertln()
            self._screen.move(9, 0)

        elif state == 23:
            self.set_state(21, False)
            self._display_progr(8, 9)
            self._screen.move(10, 0)
            self._screen.insertln()
            self._screen.move(10, 0)

        elif state == 24:
            self.set_state(21, False)
            self._display_progr(8, 10)
            self._screen.move(11, 0)
            self._screen.insertln()
            self._screen.move(11, 0)

        elif state == 25:
            self.set_state(21, False)
            self._display_progr(8, 11)
            self._screen.move(12, 0)
            self._screen.insertln()
            self._screen.move(12, 0)

        elif state == 26:
            self.set_state(21, False)
            self._display_progr(8, 12)
            self._screen.move(13, 0)
            self._screen.insertln()
            self._screen.move(13, 0)

        elif state == 27:
            self.set_state(0, False)
            self._insert_text(9, [
                '  - Generate educational requirements',
                '  - Generate job location',
                '  - Generate employment type',
                '  - Generate base salary',
                '  - Generate currency info',
                '  - Clean base salary',
                '  - Clean employment type',
                '  - Clean work hours',
                '  - Clean educational requirements',
                '  - Merge information',
            ])

        elif state == 28:
            self.set_state(27, False)
            self._display_progr(10, 10)

        elif state == 29:
            self.set_state(27, False)
            self._display_progr(10, 11)

        elif state == 30:
            self.set_state(27, False)
            self._display_progr(10, 12)

        elif state == 31:
            self.set_state(27, False)
            self._display_progr(10, 13)

        elif state == 32:
            self.set_state(27, False)
            self._display_progr(10, 14)

        elif state == 33:
            self.set_state(27, False)
            self._display_progr(10, 15)

        elif state == 34:
            self.set_state(27, False)
            self._display_progr(10, 16)

        elif state == 35:
            self.set_state(27, False)
            self._display_progr(10, 17)

        elif state == 36:
            self.set_state(27, False)
            self._display_progr(10, 18)

        elif state == 37:
            self.set_state(27, False)
            self._display_progr(10, 19)

        elif state == 38:
            self.set_state(0, False)
            self._insert_text(11, [
                '  - Merge all data',
                '  - Save data to SQL',
                '  - Export final data',
            ])

        elif state == 39:
            self.set_state(38, False)
            self._display_progr(12, 12)

        elif state == 40:
            self.set_state(38, False)
            self._display_progr(12, 13)
            self._screen.move(14, 0)
            self._screen.insertln()
            self._screen.move(14, 0)

        elif state == 41:
            self.set_state(38, False)
            self._display_progr(12, 14)

        if refresh:
            self._screen.refresh()

    def _display_progr(self, start, current):
        for i in range(current - start):
            self._screen.addstr(start + i, 2, '✓')
        self._screen.addstr(current, 2, '~')

    def _insert_text(self, start, text):
        self._screen.move(start, 0)
        for i in range(len(text) + 1):
            self._screen.insertln()
        for i in range(len(text)):
            self._screen.addstr(start + i + 1, 0, text[i])
