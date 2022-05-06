import streamlit as st


page_config = st.set_page_config(
    page_title="Page Title",
)


def get_session_state():
    # Initialize session state
    if not st.session_state.get('INIT', False):
        st.session_state['button_click_count'] = 0

    st.session_state['INIT'] = True
    return st.session_state


def main():
    session_state = get_session_state()
    st.write("# Simple Streamlit Demo")

    if st.button("Click Me"):
        session_state["button_click_count"] += 1
        clicks = session_state["button_click_count"]
        st.write(
            f"This button has been clicked {clicks} times in this session."
        )


if __name__ == '__main__':
    main()
